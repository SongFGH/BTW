"""
Bounded Treewidth (BTW) Sampling for Efficient Inference
"""
import os
from collections import defaultdict
from typing import TextIO

import numpy as np


class Graph:
    """Class of a general undirected graph."""

    def __init__(self):
        """Class initializer."""
        self.edges = []

    def degree(self) -> list:
        """Return the node degree."""
        return [len(neighbors) for neighbors in self.edges]

    def num_nodes(self) -> int:
        """Count the number of nodes."""
        return len(self.edges)

    def num_edges(self) -> int:
        """Count the number of (undirected) edges."""
        cnt = 0
        for nodes in self.edges:
            cnt += len(nodes)
        return cnt // 2

    def add_node(self, node: int):
        """Add a node to the graph."""
        while node >= self.num_nodes():
            self.edges.append(set())

    def from_csv(self, file: str):
        """Read edges from a CSV file."""
        with open(file) as f:
            for line in f:
                words = line.split(',')
                src = int(words[0])
                dest = int(words[1])
                self.add_edge(src, dest)

    def set_edges(self, edges: list):
        """Set edges from a list."""
        self.edges = edges

    def add_edge(self, src: int, dest: int):
        """Add an undirected edge from a pair of vertices."""
        self.add_node(max(src, dest))
        self.edges[src].add(dest)
        self.edges[dest].add(src)

    def contains(self, src: int, dest: int) -> bool:
        """Check if an edge is contained in the graph."""
        return src < self.num_nodes() and dest in self.edges[src]

    def nodes(self) -> set:
        """Return the set of all nodes."""
        return set(range(self.num_nodes()))

    def get_neighbors(self, nodes: int or set) -> set:
        """Return the set of neighbors of a given node(s)."""
        if type(nodes) == int:
            return self.edges[nodes]
        elif type(nodes) == set:
            neighbors = set()
            for n in nodes:
                neighbors.update(self.edges[n])
            return neighbors

    def copy(self):
        """Clone the current graph."""
        graph = Graph()
        graph.set_edges(self.copy_edges())
        return graph

    def copy_edges(self) -> list:
        """Clone all the edges."""
        return [set(neighbors) for neighbors in self.edges]

    def get_edges_by_list(self) -> list:
        """Return the edges as a simple list."""
        edge_list = []
        for src, edges in enumerate(self.edges):
            for dest in edges:
                if src < dest:
                    edge_list.append((src, dest))
        return edge_list

    def get_multiple_neighbors(self, nodes: set) -> set:
        """Return the set of neighbors of multiple given nodes."""

    def add_node_with_connections(self, node: int, neighbors: set):
        """Add a node with connecting it to existing nodes."""
        self.add_node(node)
        self.edges[node] = neighbors
        for n in neighbors:
            self.edges[n].add(node)

    def to_csv(self, file: str):
        """Store the edges as a CSV file."""
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w') as f:
            for src, neighbors in enumerate(self.edges):
                for dest in neighbors:
                    if dest > src:
                        f.write('{},{}\n'.format(src, dest))

    def sample_by_submatrix(self, num_nodes: int) -> 'Graph':
        """Sample the edges by a principal submatrix."""
        num_nodes = min(self.num_nodes(), num_nodes)
        new_edges = [set() for _ in range(num_nodes)]
        for src in range(num_nodes):
            for dest in self.edges[src]:
                if dest < num_nodes:
                    new_edges[src].add(dest)
        graph = Graph()
        graph.set_edges(new_edges)
        return graph

    def mask_evidence(self, labels: dict) -> 'Graph':
        """Mask evidence and return a new graph."""
        new_graph = self.copy()
        new_edges = new_graph.edges
        for node in labels:
            if node < self.num_nodes():
                for neighbor in self.edges[node]:
                    if neighbor in new_edges[node]:
                        new_edges[node].remove(neighbor)
                        new_edges[neighbor].remove(node)
        return new_graph

    def _count_pairs(self, labels: dict, num_states: int) -> np.ndarray:
        """Count connected pairs of observed nodes."""
        matrix = np.ones((num_states, num_states))
        for n1 in labels:
            if n1 >= self.num_nodes():
                continue
            l1 = labels[n1]
            for n2 in self.edges[n1]:
                if n2 not in labels:
                    continue
                l2 = labels[n2]
                matrix[l1][l2] += 1
                matrix[l2][l1] += 1
        return matrix

    def potential_matrix(self, labels: dict, divider: float) -> np.ndarray:
        """Generate and return a potential matrix."""
        num_states = max(labels.values()) + 1
        matrix = self._count_pairs(labels, num_states)
        matrix /= matrix.sum()
        matrix += (divider - 1) / num_states ** 2
        matrix /= matrix.sum()
        return matrix

    def to_priors(self, labels: dict, potential: np.ndarray) -> np.ndarray:
        """Return the priors of nodes based on the potential matrix."""
        num_nodes = self.num_nodes()
        num_classes = max(labels.values()) + 1
        priors = np.full((num_nodes, num_classes), 1 / num_classes)
        for node, label in labels.items():
            if node < num_nodes:
                for neighbor in self.edges[node]:
                    priors[neighbor] *= potential[label]
                    priors[neighbor] /= priors[neighbor].sum()
        return priors

    # noinspection PyUnresolvedReferences
    def count_components(self) -> list:
        """Count the sizes of connected components."""
        nodes = self.nodes()
        components = []
        while nodes:
            queue = [next(iter(nodes))]
            components.append(0)
            while queue:
                node = queue.pop(0)
                if node in nodes:
                    queue.extend(list(self.edges[node].intersection(nodes)))
                    nodes.remove(node)
                    components[-1] += 1
        return sorted(components, reverse=True)

    def _nums_neighbors(self) -> np.ndarray:
        """Return the numbers of neighbors as an array."""
        nums_neighbors = [len(self.edges[i]) for i in range(self.num_nodes())]
        return np.array(nums_neighbors, dtype=int)

    def _connect_neighbors(self, node: int):
        """Connect all the neighbors of every node."""
        neighbors = self.edges[node]
        for node1 in neighbors:
            for node2 in neighbors:
                if node1 < node2:
                    self.edges[node1].add(node2)
                    self.edges[node2].add(node1)

    def _pop_clique(self, nums_neighbors: np.ndarray, node: int) -> set:
        """Pop a clique and update the numbers of neighbors."""
        neighbors = self.edges[node]
        for neighbor in neighbors:
            self.edges[neighbor].remove(node)
            nums_neighbors[neighbor] -= 1
        self.edges[node] = set()
        neighbors.add(node)
        nums_neighbors[node] = np.iinfo(int).max
        return neighbors

    @staticmethod
    def _check_treewidth(clique: set, max_treewidth: int):
        """Check the size of a clique and raise a MemoryError."""
        if max_treewidth is not None:
            if len(clique) > max_treewidth + 1:
                raise MemoryError()

    def triangulate(self, max_treewidth: int = None) -> list:
        """Triangulate the graph and return the list of cliques."""
        triangulated = self.copy()
        nums_neighbors = triangulated._nums_neighbors()
        cliques = []
        for _ in range(self.num_nodes()):
            node = int(np.argmin(nums_neighbors))
            triangulated._connect_neighbors(node)
            new_clique = triangulated._pop_clique(nums_neighbors, node)
            self._check_treewidth(new_clique, max_treewidth)
            cliques.append(new_clique)
        return cliques


class _DisjointSet:
    """Class of a disjoint-set data structure."""

    def __init__(self, num_nodes: int):
        """Class initializer."""
        self.parents = []
        self.ranks = []
        for node in range(num_nodes):
            self.parents.append(node)
            self.ranks.append(0)

    def find(self, node: int) -> int:
        """Find the parent of a given node."""
        parent = self.parents[node]
        if parent != node:
            self.parents[node] = self.find(parent)
        return self.parents[node]

    def union(self, node1: int, node2: int):
        """Union two nodes."""
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            rank1 = self.ranks[root1]
            rank2 = self.ranks[root2]
            if rank1 > rank2:
                self.parents[root2] = root1
            elif rank1 < rank2:
                self.parents[root1] = root2
            else:
                self.parents[root1] = root2
                self.ranks[root2] += 1


class _MaxCliques:
    """Class that contains maximal cliques."""

    @staticmethod
    def _group_cliques_by_size(cliques: list) -> list:
        """Group cliques by their sizes."""
        result = defaultdict(lambda: [])
        for clique in cliques:
            result[len(clique)].append(clique)
        return sorted(result.items(), key=lambda x: x[0])

    def _find_maximal_cliques(self, cliques: list) -> list:
        """Find the maximal cliques given a list of cliques."""
        cliques_by_size = self._group_cliques_by_size(cliques)
        prev_cliques = []
        for size, curr_cliques in cliques_by_size:
            for c1 in curr_cliques:
                prev_cliques = [c for c in prev_cliques if not c.issubset(c1)]
            prev_cliques.extend(curr_cliques)
        return prev_cliques

    def __init__(self, cliques: list):
        """Class initializer."""
        self.cliques = self._find_maximal_cliques(cliques)

    def _form_complete_graph(self) -> list:
        """Form the complete graph given maximal cliques."""
        edges = []
        for i1, c1 in enumerate(self.cliques):
            for i2, c2 in enumerate(self.cliques):
                if i1 < i2:
                    edges.append((i1, i2, len(c1.intersection(c2))))
        edges = [e for e in edges if e[2] > 0]
        edges = sorted(edges, key=lambda x: x[2], reverse=True)
        return [(src, dest) for src, dest, weight in edges]

    def form_junction_tree(self):
        """Form a junction tree."""
        all_edges = self._form_complete_graph()
        disjoint_set = _DisjointSet(len(self.cliques))
        edges = defaultdict(lambda: set())
        for node1, node2 in all_edges:
            if disjoint_set.find(node1) != disjoint_set.find(node2):
                disjoint_set.union(node1, node2)
                edges[node2].add(node1)
                edges[node1].add(node2)
        return edges


class JunctionTree:
    """Class of a junction tree (or a clique tree)."""

    def __init__(self):
        """Class initializer."""
        self.cliques = []
        self.edges = []

    def copy(self) -> 'JunctionTree':
        """Copy the current junction tree."""
        junctree = JunctionTree()
        junctree.cliques = [c.copy() for c in self.cliques]
        junctree.edges = [s.copy() for s in self.edges]
        return junctree

    def _remove_node_from_cliques(self, node: int) -> set:
        """Remove a node from the cliques."""
        removed_cliques = set()
        for idx, clique in enumerate(self.cliques):
            if node in clique:
                clique.remove(node)
                removed_cliques.add(idx)
        return removed_cliques

    def _edges_as_list(self) -> list:
        """Return the edges as a list."""
        edges = []
        for src, neighbors in enumerate(self.edges):
            for dest in neighbors:
                if src < dest:
                    edges.append((src, dest))
        return edges

    def remove_node(self, node: int):
        """Remove a node from the current tree."""
        removed_cliques = self._remove_node_from_cliques(node)
        for src, dest in self._edges_as_list():
            if src in removed_cliques and dest in removed_cliques:
                src_clique = self.cliques[src]
                dst_clique = self.cliques[dest]
                if len(src_clique.intersection(dst_clique)) == 0:
                    self.edges[src].remove(dest)
                    self.edges[dest].remove(src)

    def _add_node(self, clique: int, node: int):
        """Add a clique to the junction tree."""
        while clique >= len(self.cliques):
            self.cliques.append(set())
            self.edges.append(set())
        self.cliques[clique].add(node)

    def add_edge(self, src: int, dest: int):
        """Add an edge to the junction tree."""
        self.edges[src].add(dest)
        self.edges[dest].add(src)

    def from_csv(self, file: str):
        """Read edges from a CSV file."""
        with open(file) as f:
            for line in f:
                words = line.split(',')
                line_type = words[0]
                if line_type == 'c':
                    self._add_node(int(words[1]), int(words[2]))
                elif line_type == 'e':
                    self.add_edge(int(words[1]), int(words[2]))

    def add_clique(self, clique: set, clique_id: int = None):
        """Add a clique to the junction tree."""
        self.cliques.append(set(clique))
        self.edges.append(set())
        if clique_id is not None:
            self.add_edge(len(self.cliques) - 1, clique_id)

    def treewidth(self) -> int:
        """Return the treewidth."""
        return max(len(clique) for clique in self.cliques) - 1

    def num_cliques(self) -> int:
        """Return the number of cliques."""
        return len(self.cliques)

    def last_clique(self) -> (int, set):
        """Return the last clique and its index."""
        return len(self.cliques) - 1, self.cliques[-1]

    def get(self, idx: int) -> set:
        """Return the clique of a given index."""
        return self.cliques[idx]

    def _write_cliques(self, file: TextIO):
        """Write the cliques to a file."""
        for idx, clique in enumerate(self.cliques):
            for node in clique:
                file.write('{},{},{}\n'.format('c', idx, node))

    def _write_edges(self, file: TextIO):
        """Write the edges to a file."""
        for src, neighbors in enumerate(self.edges):
            for dest in neighbors:
                if dest > src:
                    file.write('{},{},{}\n'.format('e', src, dest))

    def to_csv(self, file: str):
        """Store the edges as a CSV file."""
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w') as f:
            self._write_cliques(f)
            self._write_edges(f)

    def set_triangulated(self, graph: Graph, max_treewidth: int = None):
        """Set the cliques and edges by triangulating a given graph."""
        cliques = graph.triangulate(max_treewidth)
        max_cliques = _MaxCliques(cliques)
        self.cliques = max_cliques.cliques
        self.edges = max_cliques.form_junction_tree()

    def common_nodes(self, cid1: int, cid2: int):
        """Return the common nodes appearing in both cliques."""
        nodes1 = self.cliques[cid1]
        nodes2 = self.cliques[cid2]
        return nodes1.intersection(nodes2)
