import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torch.optim
import torch_geometric
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from experiments.zinc.model import (
    build_homp_model,
    build_sequential_subcomplex_model,
)
from data.complex_dataset import ComplexDataset, construct_datasets
from utils.training import train_loop
from utils.utils import build_run_tag


# =========================
# Euler-characteristic logic
# =========================

def _safe_num_cells(data, rank: int) -> int:
    """
    Return the number of cells in rank `rank`.
    Assumes the repo stores rank-r features in attributes x_0, x_1, x_2, ...
    """
    key = f"x_{rank}"
    if not hasattr(data, key):
        return 0
    x = getattr(data, key)
    if x is None:
        return 0
    if torch.is_tensor(x):
        return int(x.size(0))
    return 0


def compute_euler_characteristic(data) -> int:
    """
    Euler characteristic:
        chi = n0 - n1 + n2 - n3 + ...
    For the lifted ZINC setting used in the repo, ranks usually stop at 2.
    """
    max_rank = 0
    while hasattr(data, f"x_{max_rank}"):
        max_rank += 1
    max_rank -= 1

    chi = 0
    for r in range(max_rank + 1):
        nr = _safe_num_cells(data, r)
        chi += ((-1) ** r) * nr
    return int(chi)


def add_euler_labels_inplace(dataset) -> None:
    """
    Overwrite data.y with Euler characteristic.
    Keep a few useful nuisance stats for Variant B.
    """
    for i in range(len(dataset)):
        data = dataset[i]
        n0 = _safe_num_cells(data, 0)
        n1 = _safe_num_cells(data, 1)
        n2 = _safe_num_cells(data, 2)

        chi = compute_euler_characteristic(data)

        # Main target
        data.y = torch.tensor([float(chi)], dtype=torch.float)

        # Extra stats used for nuisance matching / analysis
        data.euler_chi = torch.tensor([float(chi)], dtype=torch.float)
        data.n0 = torch.tensor([float(n0)], dtype=torch.float)
        data.n1 = torch.tensor([float(n1)], dtype=torch.float)
        data.n2 = torch.tensor([float(n2)], dtype=torch.float)


def _signature_for_variant_b(data, bin_size_n0: int = 2, bin_size_n1: int = 2) -> Tuple[int, int]:
    """
    Nuisance signature for matching graph-size-like statistics.
    In lifted ZINC, n0 and n1 are inherited from the original molecule graph,
    while n2 depends on the lifting. Since chi = n0 - n1 + n2, matching (n0, n1)
    forces the task to rely mostly on the 2-cell structure.
    """
    n0 = int(data.n0.item()) if hasattr(data, "n0") else _safe_num_cells(data, 0)
    n1 = int(data.n1.item()) if hasattr(data, "n1") else _safe_num_cells(data, 1)

    return (n0 // bin_size_n0, n1 // bin_size_n1)


def build_variant_b_indices(dataset, max_per_bucket_label: Optional[int] = None) -> List[int]:
    """
    Create a nuisance-matched subset:
      - group by coarse (n0, n1) signature
      - keep labels with at least two distinct Euler values inside the same bucket
      - optionally cap examples per (bucket, label)

    This is a practical approximation to "matched nuisance statistics".
    """
    buckets: Dict[Tuple[int, int], Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

    for idx in range(len(dataset)):
        data = dataset[idx]
        sig = _signature_for_variant_b(data)
        label = int(round(float(data.y.view(-1)[0].item())))
        buckets[sig][label].append(idx)

    selected: List[int] = []
    kept_bucket_count = 0

    for sig, label_to_indices in buckets.items():
        # only keep buckets where same coarse size signature supports >=2 different chi labels
        if len(label_to_indices) < 2:
            continue

        kept_bucket_count += 1
        for _, indices in label_to_indices.items():
            if max_per_bucket_label is None:
                selected.extend(indices)
            else:
                selected.extend(indices[:max_per_bucket_label])

    print(f"[Variant B] kept {kept_bucket_count} matched buckets")
    print(f"[Variant B] selected {len(selected)} / {len(dataset)} samples")
    return sorted(selected)


class SubsetComplexDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]


# =========================
# Model
# =========================

def get_model(cfg: DictConfig):
    if cfg.arch.model_type == "subcomplex":
        model = build_sequential_subcomplex_model(
            cin_embedding_dim=cfg.arch.cin_embedding_dim,
            subcomplex_embedding_dim=cfg.arch.subgraph_embedding_dim,
            number_cin_layers_top=cfg.arch.number_of_top_cin_layers,
            number_cin_layers_bottom=cfg.arch.number_of_bottom_cin_layers,
            number_subgraph_layers=cfg.arch.number_of_sub_complex_layers,
            max_output_rank=cfg.arch.max_output_rank,
            device=cfg.device,
            add_residual=cfg.arch.add_residual,
            use_second_conv=cfg.arch.use_second_conv,
            second_conv_type=cfg.arch.second_conv_type,
            number_of_mlp_layers=cfg.arch.number_of_mlp_layers,
            high_rank=cfg.subcomplex_high_rank,
        )
    elif cfg.arch.model_type == "homp":
        model = build_homp_model(
            number_of_blocks=cfg.arch.number_of_top_cin_layers,
            embedding_dim=cfg.arch.cin_embedding_dim,
            device=cfg.device,
            number_of_mlp_layers=cfg.arch.number_of_mlp_layers,
        )
    else:
        raise ValueError(f"Unsupported model type: {cfg.arch.model_type}")

    return model


# =========================
# Data
# =========================

def get_base_datasets(
    dataset_root: str,
    zinc_root: Optional[str],
    construct_complexes: bool,
    min_len: int,
    max_len: int,
    use_subcomplex_features: bool,
    subcomplex_low_rank: int,
    subcomplex_high_rank: int,
):
    """
    Mirrors experiments/zinc/zinc.py:
    - starts from PyG ZINC
    - constructs lifted combinatorial complexes
    - or reads precomputed ComplexDataset from disk

    The upstream repo's ZINC experiment uses construct_datasets(...) with the same
    parameters and then trains on x_0, x_1, x_2. :contentReference[oaicite:0]{index=0}
    """
    datasets = {
        name: torch_geometric.datasets.ZINC(
            split=name,
            subset=True,
            root=os.path.join(zinc_root, name) if zinc_root is not None else None,
        )
        for name in ["train", "val", "test"]
    }

    if construct_complexes:
        datasets = construct_datasets(
            root=dataset_root,
            datasets=datasets,
            min_len=min_len,
            max_len=max_len,
            use_subcomplex_features=use_subcomplex_features,
            subcomplex_low_rank=subcomplex_low_rank,
            subcomplex_high_rank=subcomplex_high_rank,
        )
    else:
        datasets = {
            name: ComplexDataset(os.path.join(dataset_root, name))
            for name in ["train", "val", "test"]
        }

    return datasets


def prepare_euler_datasets(cfg: DictConfig):
    datasets = get_base_datasets(
        dataset_root=cfg.dataset_root,
        zinc_root=cfg.zinc_root,
        construct_complexes=cfg.construct_complexes,
        min_len=cfg.min_len,
        max_len=cfg.max_len,
        use_subcomplex_features=cfg.use_subcomplex_features,
        subcomplex_low_rank=cfg.subcomplex_low_rank,
        subcomplex_high_rank=cfg.subcomplex_high_rank,
    )

    # Replace the original molecular target with Euler characteristic
    for split in ["train", "val", "test"]:
        add_euler_labels_inplace(datasets[split])

    # Variant B = nuisance-matched subset by (n0, n1) bucket
    if cfg.benchmark.variant.upper() == "B":
        for split in ["train", "val", "test"]:
            idx = build_variant_b_indices(
                datasets[split],
                max_per_bucket_label=cfg.benchmark.max_per_bucket_label,
            )
            datasets[split] = SubsetComplexDataset(datasets[split], idx)

    return datasets


def get_dataloaders(cfg: DictConfig):
    datasets = prepare_euler_datasets(cfg)

    loaders = {
        name: torch_geometric.data.DataLoader(
            datasets[name],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=(name == "train"),
            follow_batch=["x_0", "x_1", "x_2"],
        )
        for name in ["train", "val", "test"]
    }
    return loaders


# =========================
# Training
# =========================

@hydra.main(config_path=".", config_name="euler_zinc_config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    model = get_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    tag = build_run_tag(prefix=cfg.wandb.prefix, attributes=cfg)

    if cfg.wandb.log:
        wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project=cfg.wandb.project_name,
            name=tag,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    dataloaders = get_dataloaders(cfg)

    optimizer = instantiate(cfg.optimizer, model.parameters())
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    # Euler characteristic is integer-valued; L1 is a reasonable regression loss
    train_loop(
        model=model,
        train_data_loader=dataloaders["train"],
        validation_data_loader=dataloaders["val"],
        test_data_loader=dataloaders["test"],
        loss_fn=nn.L1Loss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        number_of_epochs=cfg.number_of_epochs,
        device=cfg.device,
        log_to_wandb=cfg.wandb.log,
    )

    if cfg.wandb.log:
        wandb.finish()


if __name__ == "__main__":
    main()