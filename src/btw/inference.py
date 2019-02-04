"""
Bounded Treewidth (BTW) Sampling for Efficient Inference
"""
import math
import random
import warnings

import numpy as np

from btw.graph import Graph, JunctionTree

MEMORY_LIMIT = 40 * 1024 * 1024 * 1024  # 40G


def _mask_beliefs(beliefs: np.ndarray, labels: dict):
    """Return a new set of beliefs with masking observed labels."""
    new_beliefs = np.array(beliefs)
    for node, label in labels.items():
        if node < new_beliefs.shape[0]:
            new_beliefs[node] = 0
            new_beliefs[node][label] = 1
    return new_beliefs


class BPModel:
    """Class of a loopy BP model."""

    def __init__(self,
                 threshold: float = 1e-4,
                 max_iters: int = 1000):
        """Class initializer."""
        self.threshold = threshold
        self.max_iters = max_iters
        self.num_iterations = 0

    @staticmethod
    def _init_messages(graph: Graph, num_classes: int) -> list:
        """Initialize uniform messages."""
        num_nodes = graph.num_nodes()
        messages = [{} for _ in range(num_nodes)]
        for dest, local_msgs in enumerate(messages):
            for src in graph.get_neighbors(dest):
                local_msgs[src] = np.full(num_classes, 1 / num_classes)
        return messages

    def _compute_messages(self, graph: Graph, labels: dict) -> (list, list):
        """Compute and return a pair of messages."""
        masked_graph = graph.mask_evidence(labels)
        num_classes = max(labels.values()) + 1
        old_msgs = self._init_messages(masked_graph, num_classes)
        new_msgs = self._init_messages(masked_graph, num_classes)
        return old_msgs, new_msgs

    @staticmethod
    def _compute_beliefs(priors: np.ndarray, messages: list) -> np.ndarray:
        """Compute beliefs based on current messages."""
        beliefs = priors.copy()
        for node, belief in enumerate(beliefs):
            incoming_messages = messages[node]
            for msg in incoming_messages.values():
                belief *= msg
                belief /= np.sum(belief)
        return beliefs

    @staticmethod
    def _update_messages(new_msgs: list,
                         old_msgs: list,
                         beliefs: np.ndarray,
                         potential: np.ndarray):
        """Update current messages."""
        for src, neighbors in enumerate(old_msgs):
            belief = beliefs[src]
            for dest, rev_message in neighbors.items():
                mult = belief / rev_message
                new_msg = potential.dot(mult)
                new_msg /= np.sum(new_msg)
                new_msgs[dest][src] = new_msg

    def infer(self,
              graph: Graph,
              labels: dict,
              divider: int,
              potential: np.ndarray = None) -> np.ndarray:
        """Infer the marginal distribution of unobserved variables."""
        old_msgs, new_msgs = self._compute_messages(graph, labels)
        if potential is None:
            potential = graph.potential_matrix(labels, divider)
        priors = graph.to_priors(labels, potential)
        beliefs = self._compute_beliefs(priors, new_msgs)

        diff, n_iters = np.inf, 0
        while diff >= self.threshold and n_iters < self.max_iters:
            self._update_messages(new_msgs, old_msgs, beliefs, potential)
            new_beliefs = self._compute_beliefs(priors, new_msgs)
            diff = np.max(np.abs(new_beliefs - beliefs))
            beliefs = new_beliefs
            old_msgs = new_msgs
            n_iters += 1
        self.num_iterations = n_iters

        return _mask_beliefs(beliefs, labels)


class MemoryWarning(Warning):
    """Warning raised when the memory limit is over."""
    pass


class _ProbTable:
    """Class of a probability table."""

    def __init__(self, nodes: list, num_states: int, probs: np.ndarray = None):
        """Class initializer."""
        if probs is None:
            probs = np.ones([num_states] * len(nodes), dtype=np.float64)
            probs /= probs.sum()
        self.probs = probs
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.num_states = num_states
        self.map = dict([(node, idx) for idx, node in enumerate(self.nodes)])

    def copy(self) -> '_ProbTable':
        """Copy the current probability table."""
        return _ProbTable(self.nodes, self.num_states, np.array(self.probs))

    def multiply_vars(self, nodes: list, prob: np.ndarray):
        """Multiply a joint probability of multiple variables."""
        reshape = [1] * len(self.nodes)
        for node in nodes:
            reshape[self.nodes.index(node)] = self.num_states
        self.probs *= prob.reshape(tuple(reshape))
        self.probs /= self.probs.sum()

    def multiply_table(self, table: '_ProbTable'):
        """Multiply a probability table."""
        intersection = set(self.nodes).intersection(set(table.nodes))
        marginalized = table.marginalize_vars(intersection)
        reshape = []
        for node in self.nodes:
            reshape.append(self.num_states if node in intersection else 1)
        self.probs *= marginalized.probs.reshape(tuple(reshape))
        self.probs /= self.probs.sum()

    def marginalize_var(self, node: int):
        """Marginalize by a single variable."""
        indices = list(range(self.num_nodes))
        del indices[self.map[node]]
        return self.probs.sum(axis=tuple(indices))

    def marginalize_vars(self, nodes: set):
        """Marginalize by multiple variables."""
        indices = [self.map[n] for n in self.nodes if n not in nodes]
        probs = self.probs.sum(axis=tuple(indices))
        rest_nodes = [n for n in self.nodes if n in nodes]
        probs /= probs.sum()
        return _ProbTable(rest_nodes, self.num_states, probs)


class JuncTreeModel:
    """Class of a junction tree inference model."""

    def __init__(self):
        """Class initializer."""
        self.junctree = None
        self.priors = None
        self.messages = None
        self.memory_errors = 0

    def _set_junction_tree(self, graph: Graph,
                           junctree: JunctionTree,
                           labels: dict,
                           max_treewidth: int):
        """Set a junction tree."""
        if junctree is None:
            masked_graph = graph.mask_evidence(labels)
            junctree = JunctionTree()
            junctree.set_triangulated(masked_graph, max_treewidth)
        else:
            junctree = junctree.copy()
            for node in labels:
                junctree.remove_node(node)
        self.junctree = junctree

    @staticmethod
    def _uniform_beliefs(num_nodes: int, num_states: int) -> np.ndarray:
        """Return uniform beliefs."""
        beliefs = np.random.uniform(0, 1, (num_nodes, num_states))
        return beliefs / beliefs.sum(axis=1, keepdims=True)

    @staticmethod
    def _multiply_node_priors(prob: _ProbTable,
                              priors: np.ndarray,
                              nodes: set):
        """Multiply node priors to a probability table."""
        for node in prob.nodes:
            if node in nodes:
                prob.multiply_vars([node], priors[node])
                nodes.remove(node)

    @staticmethod
    def _multiply_potentials(prob: _ProbTable,
                             potential: np.ndarray,
                             edges: list):
        """Multiply edge potentials to a probability table."""
        for node1 in prob.nodes:
            for node2 in prob.nodes:
                if node1 < node2 and node1 in edges[node2]:
                    prob.multiply_vars([node1, node2], potential)
                    edges[node2].remove(node1)
                    edges[node1].remove(node2)

    def _approximate_memory(self, num_states: int) -> int:
        """Approximate the amount of memory usage."""
        return 8 * sum(num_states ** len(c) for c in self.junctree.cliques)

    def _set_priors(self, graph: Graph,
                    labels: dict,
                    divider: float,
                    potential: np.ndarray):
        """Compute and set clique priors based on evidence."""
        if potential is None:
            potential = graph.potential_matrix(labels, divider)
        node_priors = graph.to_priors(labels, potential)

        nodes = set(range(graph.num_nodes()))
        edges = graph.mask_evidence(labels).copy_edges()
        num_states = potential.shape[0]

        self.priors = []
        for clique in self.junctree.cliques:
            clique_prior = _ProbTable(sorted(clique), num_states)
            self._multiply_node_priors(clique_prior, node_priors, nodes)
            self._multiply_potentials(clique_prior, potential, edges)
            self.priors.append(clique_prior)

    def _compute_message(self, src: int, dest: int) -> np.ndarray:
        """Compute a single message from a source to a destination."""
        message = self.priors[src].copy()
        for neighbor in self.junctree.edges[src]:
            if neighbor != dest:
                message.multiply_table(self.messages[src][neighbor])
        common_nodes = self.junctree.common_nodes(src, dest)
        return message.marginalize_vars(common_nodes)

    def _collect_messages(self, updated_nodes: set, src: int, dest: int):
        """Collect messages recursively to a root node."""
        updated_nodes.add(src)
        for neighbor in self.junctree.edges[src]:
            if neighbor != dest:
                self._collect_messages(updated_nodes, neighbor, src)
        self.messages[dest][src] = self._compute_message(src, dest)

    def _distribute_messages(self, src: int, dest: int):
        """Distribute messages recursively from a root node."""
        self.messages[dest][src] = self._compute_message(src, dest)
        for neighbor in self.junctree.edges[dest]:
            if neighbor != src:
                self._distribute_messages(dest, neighbor)

    def _propagate_messages(self, root):
        """Propagate messages by collecting and distributing them."""
        neighbors = self.junctree.edges[root]
        updated_nodes = {root}
        for neighbor in neighbors:
            self._collect_messages(updated_nodes, neighbor, root)
        for neighbor in neighbors:
            self._distribute_messages(root, neighbor)
        return updated_nodes

    def _set_messages(self):
        """Compute and set all the messages."""
        self.messages = [{} for _ in range(len(self.junctree.cliques))]
        candidates = set(i for (i, c) in enumerate(self.junctree.cliques))
        while candidates:
            root = random.sample(candidates, 1)[0]
            updated_cliques = self._propagate_messages(root)
            candidates = candidates.difference(updated_cliques)

    # noinspection PyUnresolvedReferences
    def _compute_clique_beliefs(self) -> list:
        """Compute and set the clique beliefs."""
        beliefs = []
        for clique in range(len(self.junctree.cliques)):
            belief = self.priors[clique].copy()
            for neighbor in self.junctree.edges[clique]:
                belief.multiply_table(self.messages[clique][neighbor])
            beliefs.append(belief)
        return beliefs

    def _compute_node_beliefs(self, clique_beliefs: list,
                              num_nodes: int,
                              num_states: int) -> np.ndarray:
        """Compute and return the beliefs of random variables."""
        computed_nodes = set()
        beliefs = np.zeros((num_nodes, num_states))
        for idx, clique in enumerate(self.junctree.cliques):
            for node in clique:
                if node not in computed_nodes:
                    beliefs[node] = clique_beliefs[idx].marginalize_var(node)
                    computed_nodes.add(node)
        return beliefs

    @staticmethod
    def _to_max_treewidth(num_states: int) -> int:
        """Calculate the maximum treewidth based on the number of states."""
        return math.floor((math.log(MEMORY_LIMIT) - math.log(16)) /
                          math.log(num_states)) - 1

    def infer(self, graph: Graph,
              labels: dict,
              junctree: JunctionTree = None,
              divider: int = 1,
              potential: np.ndarray = None) -> np.ndarray:
        """Infer the marginal distribution of unobserved variables."""
        num_states = max(labels.values()) + 1

        try:
            max_treewidth = self._to_max_treewidth(num_states)
            self._set_junction_tree(graph, junctree, labels, max_treewidth)
            if self._approximate_memory(num_states) > MEMORY_LIMIT / 2:
                raise MemoryError()
        except MemoryError:
            self.memory_errors += 1
            msg = 'A memory error occurred! Uniform beliefs are returned.'
            warnings.warn(msg, MemoryWarning)
            return self._uniform_beliefs(graph.num_nodes(), num_states)

        self._set_priors(graph, labels, divider, potential)
        self._set_messages()
        clique_beliefs = self._compute_clique_beliefs()
        node_beliefs = self._compute_node_beliefs(clique_beliefs,
                                                  graph.num_nodes(),
                                                  num_states)

        return _mask_beliefs(node_beliefs, labels)
