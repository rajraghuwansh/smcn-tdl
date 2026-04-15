# experiments/euler_char/dataset.py
from itertools import combinations
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from toponetx.classes import CombinatorialComplex as CCC

from data.complex_data import ComplexData

#%%
class EulerCharacteristicDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        num_pairs: int = 1000,
        min_nodes: int = 8,
        max_nodes: int = 18,
        min_cycle_len: int = 3,
        max_cycle_len: int = 5,
        seed: int = 0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.num_pairs = num_pairs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_cycle_len = min_cycle_len
        self.max_cycle_len = max_cycle_len
        self.seed = seed
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=ComplexData)
#%%
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["euler_complex_data.pt"]

    def process(self):
        rng = random.Random(self.seed)
        data_list = []

        for pair_id in range(self.num_pairs):
            graph, candidate_faces = self._generate_graph_with_faces(rng)

            num_base_faces = rng.randint(1, len(candidate_faces) - 1)
            face_indices = list(range(len(candidate_faces)))
            rng.shuffle(face_indices)

            base_faces = [candidate_faces[i] for i in face_indices[:num_base_faces]]
            extra_face = candidate_faces[face_indices[num_base_faces]]

            face_sets = [base_faces, base_faces + [extra_face]]
            for variant_id, faces in enumerate(face_sets):
                complex_data = self._build_complex_data(
                    graph=graph,
                    faces=faces,
                    pair_id=pair_id,
                    variant_id=variant_id,
                )
                data_list.append(complex_data)

        self.save(data_list, self.processed_paths[0])

    def _generate_graph_with_faces(self, rng: random.Random):
        for _ in range(256):
            num_nodes = rng.randint(self.min_nodes, self.max_nodes)
            graph = nx.cycle_graph(num_nodes)

            possible_chords = [
                edge
                for edge in combinations(range(num_nodes), 2)
                if not graph.has_edge(*edge)
                and ((edge[1] - edge[0]) % num_nodes not in (1, num_nodes - 1))
            ]
            rng.shuffle(possible_chords)

            num_chords = rng.randint(max(2, num_nodes // 4), max(3, num_nodes // 2))
            for u, v in possible_chords[:num_chords]:
                graph.add_edge(u, v)

            candidate_faces = self._get_candidate_faces(graph)
            if len(candidate_faces) >= 2:
                return graph, candidate_faces

        raise RuntimeError("Failed to generate a graph with at least two candidate 2-cells.")

    def _get_candidate_faces(self, graph: nx.Graph):
        faces = set()

        for cycle in nx.cycle_basis(graph):
            if self.min_cycle_len <= len(cycle) <= self.max_cycle_len:
                faces.add(tuple(sorted(cycle)))

        for node in graph.nodes:
            neighbors = sorted(graph.neighbors(node))
            for u, v in combinations(neighbors, 2):
                if graph.has_edge(u, v):
                    triangle = tuple(sorted((node, u, v)))
                    if self.min_cycle_len <= len(triangle) <= self.max_cycle_len:
                        faces.add(triangle)

        return sorted(faces)

    def _build_complex_data(
        self,
        graph: nx.Graph,
        faces,
        pair_id: int,
        variant_id: int,
    ) -> ComplexData:
        cc = CCC(graph)
        if faces:
            cc.add_cells_from(cells=faces, ranks=[2] * len(faces))

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        num_faces = len(faces)
        euler_char = num_nodes - num_edges + num_faces

        cell_features = [
            torch.zeros((num_nodes, 1), dtype=torch.float),
            torch.zeros((num_edges, 1), dtype=torch.float),
        ]
        y = torch.tensor([[float(euler_char)]], dtype=torch.float)

        complex_data = ComplexData.create_from_cc(
            cc,
            cell_features=cell_features,
            y=y,
            dim=2,
        )
        complex_data.compute_subcomplex_feature(low_rk=0, high_rk=2, binary_marking=False)
        complex_data.pair_id = torch.tensor([pair_id], dtype=torch.long)
        complex_data.variant_id = torch.tensor([variant_id], dtype=torch.long)
        complex_data.euler_raw = torch.tensor([float(euler_char)], dtype=torch.float)
        complex_data.num_vertices = torch.tensor([num_nodes], dtype=torch.long)
        complex_data.num_edges_graph = torch.tensor([num_edges], dtype=torch.long)
        complex_data.num_faces = torch.tensor([num_faces], dtype=torch.long)
        return complex_data
