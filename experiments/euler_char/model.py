# experiments/euler_char/model.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn

from models.embeddings import TwoCellEmbedding, ZeroEmbedding
from models.layers.merge_node import MergeNode
from models.layers.subcomplex import (
    SubComplexBroadcastLow,
    SubComplexDistanceMarkingEmbed,
    SubComplexIncidenceConv,
    SubComplexLowConv,
    SubComplexPool,
)
from models.tensor_diagram import TensorDiagram


def get_feature_embed_layer(embedding_dim: int = 94):
    return nn.ModuleDict(
        {
            "x_0": ZeroEmbedding(embedding_dim=embedding_dim, num_embeddings=1, rank=0),
            "x_1": ZeroEmbedding(embedding_dim=embedding_dim, num_embeddings=1, rank=1),
        }
    )


def get_two_cell_embed_layer(
    embedding_dim: int = 94, number_of_mlp_layers: int = 2, learned: bool = False
):
    return nn.ModuleDict(
        {
            "x_2": TwoCellEmbedding(
                embedding_dim=embedding_dim,
                number_of_mlp_layers=number_of_mlp_layers,
                learned=learned,
            )
        }
    )


def get_subcomplex_embed_layer(
    embedding_dim: int = 94,
    aggregation: str = "concatenate",
    activation: str = "relu",
    high_rank: int = 2,
):
    return nn.ModuleDict(
        {
            f"x_0_{high_rank}": MergeNode(
                [
                    SubComplexBroadcastLow(low_rank=0, high_rank=high_rank),
                    SubComplexDistanceMarkingEmbed(
                        low_rank=0, high_rank=high_rank, embed_dim=embedding_dim
                    ),
                ],
                aggregation=aggregation,
                embedding_dim=embedding_dim,
                activation=activation,
            )
        }
    )


def get_subcomplex_layer(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    number_of_mlp_layers: int = 2,
    edge_dim: int = None,
    aggregation: str = "concatenate",
    activation: str = "relu",
    conv_type: str = "gin",
    high_rank: int = 2,
):
    return nn.ModuleDict(
        {
            f"x_0_{high_rank}": MergeNode(
                [
                    SubComplexIncidenceConv(
                        low_rank=0,
                        high_rank=high_rank,
                        input_channels=input_dim,
                        output_channels=output_dim,
                        hidden_channels=hidden_dim,
                        number_of_mlp_layers=number_of_mlp_layers,
                        dropout=0.0,
                        train_eps=False,
                        activation=activation,
                    ),
                    SubComplexLowConv(
                        low_rank=0,
                        high_rank=high_rank,
                        input_channels=input_dim,
                        output_channels=output_dim,
                        hidden_channels=hidden_dim,
                        number_of_mlp_layers=number_of_mlp_layers,
                        edge_dim=edge_dim,
                        dropout=0.0,
                        train_eps=False,
                        activation=activation,
                        conv_type=conv_type,
                    ),
                ],
                aggregation=aggregation,
                embedding_dim=output_dim,
                activation=activation,
            )
        }
    )


def get_subcomplex_pooling_layer(aggregation: str = "sum", high_rank: int = 2):
    return nn.ModuleDict(
        {
            "x_0": SubComplexPool(
                low_rank=0, high_rank=high_rank, return_low_rank=True, aggregation=aggregation
            ),
            f"x_{high_rank}": SubComplexPool(
                low_rank=0,
                high_rank=high_rank,
                return_low_rank=False,
                aggregation=aggregation,
            ),
        }
    )


def build_euler_model(
    embedding_dim: int,
    number_subgraph_layers: int,
    device: str,
    output_ranks=None,
    number_of_mlp_layers: int = 2,
    dropout: float = 0.0,
    in_dropout: float = 0.0,
    residual: bool = False,
    activation: str = "relu",
    high_rank: int = 2,
) -> TensorDiagram:
    if output_ranks is None:
        output_ranks = [0, 2]

    number_subgraph_layers += 2
    embedding_dim = round(embedding_dim * 1.5)

    layers = nn.ModuleList(
        [
            get_feature_embed_layer(embedding_dim=embedding_dim),
            get_two_cell_embed_layer(
                embedding_dim=embedding_dim,
                number_of_mlp_layers=number_of_mlp_layers,
            ),
            get_subcomplex_embed_layer(
                embedding_dim=embedding_dim,
                aggregation="concatenate",
                activation=activation,
                high_rank=high_rank,
            ),
        ]
    )

    residual_list = [False, False, False]

    for _ in range(number_subgraph_layers):
        layers.append(
            get_subcomplex_layer(
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                hidden_dim=embedding_dim,
                number_of_mlp_layers=number_of_mlp_layers,
                edge_dim=embedding_dim,
                aggregation="concatenate",
                activation=activation,
                conv_type="gin",
                high_rank=high_rank,
            )
        )
        residual_list.append(residual)

    layers.append(
        get_subcomplex_pooling_layer(
            aggregation="concatenate",
            high_rank=high_rank,
        )
    )
    residual_list.append(False)

    dropout_list = [in_dropout] * 2 + (len(layers) - 2) * [dropout]

    return TensorDiagram(
        layers=layers,
        embedding_dim=embedding_dim,
        output_dim=1,
        output_ranks=output_ranks,
        device=device,
        dropout_list=dropout_list,
        final_dropout=dropout,
        aggregation="sum",
        residuals=residual_list,
        activation=activation,
        zinc_head=False,
    )


class EulerSMCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = build_euler_model(
            embedding_dim=config.hidden_dim,
            number_subgraph_layers=config.num_layers,
            device=getattr(config, "device", "cpu"),
            output_ranks=getattr(config, "output_ranks", [0, 2]),
            number_of_mlp_layers=getattr(config, "number_of_mlp_layers", 2),
            dropout=getattr(config, "dropout", 0.0),
            in_dropout=getattr(config, "in_dropout", 0.0),
            residual=getattr(config, "residual", False),
            activation=getattr(config, "activation", "relu"),
            high_rank=getattr(config, "high_rank", 2),
        )

    def forward(self, data):
        return self.model(data)
