"""
Bounded Treewidth (BTW) Sampling for Efficient Inference
"""
import random

import numpy as np

from btw.graph import Graph, JunctionTree


class _Element:
    """Class of a tuple that is contained in a max-heap."""

    def __init__(self, node: int):
        """Class initializer."""
        self.node = node
        self.score = np.random.uniform(0, 1)
        self.clique_id = None

    def __lt__(self, other: '_Element') -> bool:
        """Operator of 'less than.'"""
        return self.score > other.score

    def set(self, score: int, clique_id: int):
        """Set the score and clique ID."""
        self.score = score
        self.clique_id = clique_id

    def get(self) -> (int, float, int):
        """Return the variables."""
        return self.node, self.score, self.clique_id


# noinspection PyTypeChecker
class _MaxHeap:
    """Class of a max-heap containing the scores."""

    def __init__(self, num_nodes: int, initial_nodes: set):
        """Class initializer."""
        active_nodes = list(set(range(num_nodes)).difference(initial_nodes))
        self._initialize(num_nodes, active_nodes)
        self._heapify()

    def _reposition(self, pos: int, elem: _Element):
        """Set an element in a given position."""
        self.heap[pos] = elem
        self.positions[elem.node] = pos

    def _sift_down(self, start_pos: int, pos: int):
        """Sift down."""
        new_item = self.heap[pos]
        while pos > start_pos:
            parent_pos = (pos - 1) >> 1
            parent = self.heap[parent_pos]
            if new_item < parent:
                self._reposition(pos, parent)
                pos = parent_pos
                continue
            break
        self._reposition(pos, new_item)

    def _sift_up(self, start_pos: int, pos: int):
        """Sift up."""
        end_pos = len(self.heap)
        new_item = self.heap[pos]
        child_pos = 2 * pos + 1
        while child_pos < end_pos:
            right_pos = child_pos + 1
            if right_pos < end_pos:
                if not self.heap[child_pos] < self.heap[right_pos]:
                    child_pos = right_pos
            self._reposition(pos, self.heap[child_pos])
            pos = child_pos
            child_pos = 2 * pos + 1
        self._reposition(pos, new_item)
        self._sift_down(start_pos, pos)

    def _initialize(self, num_nodes: int, active_nodes: list):
        """Initialize the positions of active nodes."""
        self.positions = [None] * num_nodes
        self.heap = [_Element(i) for i in active_nodes]
        for idx, node in enumerate(active_nodes):
            self.positions[int(node)] = idx

    def _heapify(self):
        """Initialize a heap of given active nodes."""
        for i in reversed(range(len(self.heap) // 2)):
            self._sift_up(i, i)

    def contain(self, node: int):
        """Check whether a given node is activated."""
        return self.positions[node] is not None

    def get_score(self, node: int):
        """Return the score of a given node."""
        return self.heap[int(self.positions[node])].score

    def set_element(self, node: int, score: int, clique_id: int):
        """Set an element."""
        pos = int(self.positions[node])
        self.heap[pos].set(score, clique_id)
        self._sift_up(0, pos)

    def pop(self):
        """Pop the element with the maximal score."""
        last_elem = self.heap.pop()
        if not self.heap:
            self.positions[last_elem.node] = None
            return last_elem.get()
        return_item = self.heap[0]
        self._reposition(0, last_elem)
        self._sift_up(0, 0)
        self.positions[return_item.node] = None
        return return_item.get()

    @staticmethod
    def _compute_score(graph: Graph, weights: list, clique: set, node: int):
        """Compute a score between a node and a clique."""
        commons = graph.get_neighbors(node).intersection(clique)
        return len(commons) + sum(weights[node][c] for c in commons)

    def update_scores(self, graph: Graph,
                      weights: list,
                      k_tree: JunctionTree,
                      new_nodes: int or set = None):
        """Update the scores."""
        clique_id, clique = k_tree.last_clique()
        if new_nodes is None:
            new_nodes = clique
        targets = graph.get_neighbors(new_nodes)
        for target in targets:
            if self.contain(target):
                old_score = self.get_score(target)
                new_score = self._compute_score(graph, weights, clique, target)
                if new_score > old_score:
                    self.set_element(target, new_score, clique_id)


def _add_node(graph: Graph, candidates: set, chosen_nodes: set):
    """Add an additional node to the initial graph."""
    new_node = random.sample(candidates, 1)[0]
    chosen_nodes.add(new_node)
    neighbors = graph.get_neighbors(new_node).difference(chosen_nodes)
    candidates.remove(new_node)
    candidates.update(neighbors)


def _initialize_nodes(graph: Graph, bound: int) -> set:
    """Initializes the set of k + 1 chosen nodes."""
    chosen_nodes = set()
    candidates = {random.randint(0, graph.num_nodes() - 1)}
    for i in range(bound + 1):
        if not candidates:
            tmp = graph.nodes().difference(chosen_nodes)
            candidates = {random.sample(tmp, 1)[0]}

        new_node = random.sample(candidates, 1)[0]
        chosen_nodes.add(new_node)
        neighbors = graph.get_neighbors(new_node).difference(chosen_nodes)
        candidates.remove(new_node)
        candidates.update(neighbors)
    return chosen_nodes


def _initialize_subgraph(graph: Graph, initial_nodes: set) -> Graph:
    """Initialize the induced subgraph of the initial nodes."""
    subgraph = Graph()
    for node in initial_nodes:
        neighbors = graph.get_neighbors(node).intersection(initial_nodes)
        subgraph.add_node(node)
        subgraph.edges[node] = neighbors
    return subgraph


def _compute_weights(graph: Graph) -> list:
    """Compute weights of the edges."""
    max_degree = max(len(neighbors) for neighbors in graph.edges)
    weights = [{} for _ in range(graph.num_nodes())]
    for src, neighbors in enumerate(graph.edges):
        for dest in neighbors:
            if src < dest:
                weight = np.random.uniform(0, 1) / max_degree
                weights[src][dest] = weight
                weights[dest][src] = weight
    return weights


def _adjust_clique(graph: Graph, clique: set, node: int) -> set:
    """Replace a random node in a given clique as a new one."""
    neighbors = graph.get_neighbors(node)
    targets = set.difference(clique, neighbors)
    if targets:
        target = random.sample(targets, 1)[0]
    else:
        target = random.sample(clique, 1)[0]
    new_clique = set(clique)
    new_clique.remove(target)
    new_clique.add(node)
    return new_clique


def sample(graph: Graph,
           bound: int,
           optimize: bool = True) -> (Graph, JunctionTree):
    """Sample a subgraph using BTW."""
    initial_nodes = _initialize_nodes(graph, bound)
    subgraph = _initialize_subgraph(graph, initial_nodes)
    candidates = set.difference(graph.nodes(), initial_nodes)

    ktree = JunctionTree()
    ktree.add_clique(initial_nodes)

    weights = _compute_weights(graph)
    heap = _MaxHeap(graph.num_nodes(), initial_nodes)
    heap.update_scores(graph, weights, ktree)

    while candidates:
        new_node, _, clique_id = heap.pop()
        if clique_id is None:
            clique_id = np.random.randint(0, ktree.num_cliques())
        new_clique = _adjust_clique(graph, ktree.get(clique_id), new_node)

        neighbors = graph.get_neighbors(new_node).intersection(new_clique)
        subgraph.add_node_with_connections(new_node, neighbors)
        ktree.add_clique(new_clique, clique_id)

        target = new_node if optimize else new_clique
        heap.update_scores(graph, weights, ktree, target)

        candidates.remove(new_node)

    return subgraph, ktree
