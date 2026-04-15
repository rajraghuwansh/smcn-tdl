from pathlib import Path
import random
import sys
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from toponetx.classes import CombinatorialComplex as CCC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.complex_data import ComplexData


GridSize = Tuple[int, int]


class OrientabilityDataset(InMemoryDataset):
    """
    Matched family of orientable/non-orientable square-cell surfaces.

    For each pair ID we generate the same multiset of component sizes twice:
    - orientable sample: all components are tori
    - non-orientable sample: exactly one matched component is replaced by a Klein bottle

    Both members of a pair have identical numbers of vertices, edges and faces and the
    same local incidence degrees. They differ only through the global gluing pattern.
    """

    def __init__(
        self,
        root: str,
        num_pairs: int = 256,
        min_components: int = 1,
        max_components: int = 3,
        min_grid_size: int = 5,
        max_grid_size: int = 9,
        number_of_permuted_copies: int = 8,
        low_rk: int = 0,
        high_rk: int = 2,
        seed: int = 0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.num_pairs = num_pairs
        self.min_components = min_components
        self.max_components = max_components
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
        self.number_of_permuted_copies = number_of_permuted_copies
        self.low_rk = low_rk
        self.high_rk = high_rk
        self.seed = seed
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=ComplexData)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["orientability_dataset.pt"]

    def process(self):
        rng = random.Random(self.seed)
        data_list = self._create_data_list(rng)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def _create_data_list(self, rng: random.Random) -> List[ComplexData]:
        data_list: List[ComplexData] = []
        for pair_id in range(self.num_pairs):
            component_sizes = self._sample_component_sizes(rng)
            twisted_component = rng.randrange(len(component_sizes))

            orientable_cc = build_surface_family(
                component_sizes=component_sizes,
                component_types=["torus"] * len(component_sizes),
            )
            nonorientable_types = ["torus"] * len(component_sizes)
            nonorientable_types[twisted_component] = "klein"
            nonorientable_cc = build_surface_family(
                component_sizes=component_sizes,
                component_types=nonorientable_types,
            )

            orientable_data = self._finalize_complex(
                cc=orientable_cc,
                pair_id=pair_id,
                label=1.0,
                component_sizes=component_sizes,
                twisted_component=-1,
            )
            nonorientable_data = self._finalize_complex(
                cc=nonorientable_cc,
                pair_id=pair_id,
                label=0.0,
                component_sizes=component_sizes,
                twisted_component=twisted_component,
            )

            self._assert_pair_match(orientable_data, nonorientable_data)
            data_list.extend(self._expand_with_permutations(orientable_data))
            data_list.extend(self._expand_with_permutations(nonorientable_data))
        return data_list

    def _sample_component_sizes(self, rng: random.Random) -> List[GridSize]:
        num_components = rng.randint(self.min_components, self.max_components)
        component_sizes = []
        for _ in range(num_components):
            n = rng.randint(self.min_grid_size, self.max_grid_size)
            m = rng.randint(self.min_grid_size, self.max_grid_size)
            component_sizes.append((n, m))
        component_sizes.sort()
        return component_sizes

    def _finalize_complex(
        self,
        cc: CCC,
        pair_id: int,
        label: float,
        component_sizes: Sequence[GridSize],
        twisted_component: int,
    ) -> ComplexData:
        cell_features = [
            torch.zeros((cc.number_of_nodes(), 1), dtype=torch.float),
        ]
        data = ComplexData.create_from_cc(
            cc=cc,
            cell_features=cell_features,
            dim=2,
            y=torch.tensor([[label]], dtype=torch.float),
        )
        data.compute_subcomplex_feature(
            low_rk=self.low_rk,
            high_rk=self.high_rk,
            binary_marking=False,
        )

        num_faces = getattr(data, "x_2").size(0)
        component_tensor = torch.tensor(component_sizes, dtype=torch.long).reshape(-1)
        data.pair_id = torch.tensor([pair_id], dtype=torch.long)
        data.is_orientable = torch.tensor([int(label)], dtype=torch.long)
        data.twisted_component = torch.tensor([twisted_component], dtype=torch.long)
        data.component_sizes = component_tensor
        data.num_components = torch.tensor([len(component_sizes)], dtype=torch.long)
        data.num_vertices_total = torch.tensor([cc.number_of_nodes()], dtype=torch.long)
        data.num_edges_total = torch.tensor([cc.number_of_cells(1)], dtype=torch.long)
        data.num_faces_total = torch.tensor([num_faces], dtype=torch.long)
        return data

    def _expand_with_permutations(self, data: ComplexData) -> List[ComplexData]:
        copies = [data]
        if self.number_of_permuted_copies > 1:
            copies = data.get_permuted_copies(self.number_of_permuted_copies)

        expanded = []
        for copy_id, copy in enumerate(copies):
            copy.compute_subcomplex_feature(
                low_rk=self.low_rk,
                high_rk=self.high_rk,
                binary_marking=False,
            )
            copy.y = data.y.clone()
            copy.pair_id = data.pair_id.clone()
            copy.is_orientable = data.is_orientable.clone()
            copy.twisted_component = data.twisted_component.clone()
            copy.component_sizes = data.component_sizes.clone()
            copy.num_components = data.num_components.clone()
            copy.num_vertices_total = data.num_vertices_total.clone()
            copy.num_edges_total = data.num_edges_total.clone()
            copy.num_faces_total = data.num_faces_total.clone()
            copy.copy_id = torch.tensor([copy_id], dtype=torch.long)
            expanded.append(copy)
        return expanded

    @staticmethod
    def _assert_pair_match(first: ComplexData, second: ComplexData):
        assert first.num_vertices_total.item() == second.num_vertices_total.item()
        assert first.num_edges_total.item() == second.num_edges_total.item()
        assert first.num_faces_total.item() == second.num_faces_total.item()
        assert first.num_components.item() == second.num_components.item()


def build_surface_family(
    component_sizes: Iterable[GridSize],
    component_types: Sequence[str],
) -> CCC:
    graph = nx.Graph()
    all_faces = []
    node_offset = 0

    for (rows, cols), component_type in zip(component_sizes, component_types):
        component_graph, component_faces = build_surface_component(
            rows=rows,
            cols=cols,
            manifold=component_type,
        )
        relabel = {node: node + node_offset for node in component_graph.nodes()}
        component_graph = nx.relabel_nodes(component_graph, relabel)
        graph = nx.compose(graph, component_graph)
        all_faces.extend(
            [tuple(node + node_offset for node in face) for face in component_faces]
        )
        node_offset += rows * cols

    cc = CCC(graph)
    cc.add_cells_from(cells=all_faces, ranks=[2] * len(all_faces))
    return cc


def build_surface_component(rows: int, cols: int, manifold: str) -> Tuple[nx.Graph, List[Tuple[int, ...]]]:
    assert manifold in {"torus", "klein"}
    graph = nx.Graph()
    graph.add_nodes_from(range(rows * cols))

    def vertex_id(i: int, j: int) -> int:
        i_wrap = i % rows
        j_wrap = j % cols
        if manifold == "klein" and (j // cols) % 2 != 0:
            i_wrap = rows - 1 - i_wrap
        return i_wrap * cols + j_wrap

    faces = []
    for i in range(rows):
        for j in range(cols):
            v00 = vertex_id(i, j)
            v01 = vertex_id(i, j + 1)
            v11 = vertex_id(i + 1, j + 1)
            v10 = vertex_id(i + 1, j)

            graph.add_edge(v00, v01)
            graph.add_edge(v01, v11)
            graph.add_edge(v11, v10)
            graph.add_edge(v10, v00)
            faces.append((v00, v01, v11, v10))

    return graph, faces


def get_orientability_pair(
    component_sizes: Sequence[GridSize] | None = None,
    grid_size: int | None = None,
    twisted_component: int = 0,
    low_rk: int = 0,
    high_rk: int = 2,
) -> Tuple[ComplexData, ComplexData]:
    if component_sizes is None:
        if grid_size is None:
            grid_size = 8
        component_sizes = [(grid_size, grid_size)]

    orientable_cc = build_surface_family(
        component_sizes=component_sizes,
        component_types=["torus"] * len(component_sizes),
    )
    nonorientable_types = ["torus"] * len(component_sizes)
    nonorientable_types[twisted_component] = "klein"
    nonorientable_cc = build_surface_family(
        component_sizes=component_sizes,
        component_types=nonorientable_types,
    )

    def finalize(cc: CCC, label: float) -> ComplexData:
        cell_features = [
            torch.zeros((cc.number_of_nodes(), 1), dtype=torch.float),
        ]
        data = ComplexData.create_from_cc(
            cc=cc,
            cell_features=cell_features,
            dim=2,
            y=torch.tensor([[label]], dtype=torch.float),
        )
        data.compute_subcomplex_feature(low_rk=low_rk, high_rk=high_rk, binary_marking=False)
        return data

    return finalize(orientable_cc, 1.0), finalize(nonorientable_cc, 0.0)
