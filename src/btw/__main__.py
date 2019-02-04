"""
Bounded Treewidth (BTW) Sampling for Efficient Inference
"""
import os
import time
import warnings
from collections import defaultdict

import click
import numpy as np
from sklearn import metrics

from btw.graph import Graph, JunctionTree
from btw.inference import BPModel, JuncTreeModel
from btw.sampling import sample


def divide_nodes_by_labels(labels: dict) -> dict:
    """Divide nodes based on their labels."""
    nodes_by_labels = defaultdict(lambda: [])
    for node, label in labels.items():
        nodes_by_labels[label].append(node)
    return nodes_by_labels


def divide_nodes(labels: dict, num_folds: int, rand_seed: int = 0) -> list:
    """Divide nodes into N chunks."""
    np.random.seed(rand_seed)
    nodes_by_labels = divide_nodes_by_labels(labels)
    chunks = [[] for _ in range(num_folds)]
    for nodes in nodes_by_labels.values():
        nodes = sorted(nodes)
        np.random.shuffle(nodes)
        for i in range(num_folds):
            idx_start = (i * len(nodes)) // num_folds
            idx_end = ((i + 1) * len(nodes)) // num_folds
            chunks[i].extend(nodes[idx_start:idx_end])
    return chunks


def read_labels(file: str) -> dict:
    """Read labels from a CSV file."""
    labels = {}
    with open(file) as f:
        for line in f:
            words = line.split(',')
            node = int(words[0])
            label = int(words[1])
            labels[node] = label
    return labels


def divide_labels(labels: dict, nodes: list) -> (dict, dict):
    """Divide nodes into observed and test ones."""
    test_labels, observed_labels = {}, labels.copy()
    for node in nodes:
        test_labels[node] = labels[node]
        del observed_labels[node]
    return observed_labels, test_labels


def predict_label(beliefs: np.ndarray, node: int, num_classes: int) -> float:
    """Predict the label of a node based on its belief."""
    if node >= beliefs.shape[0]:
        return np.random.randint(0, num_classes)
    belief = beliefs[node]
    try:
        return np.random.choice(np.flatnonzero(belief == belief.max()))
    except ValueError:
        return float(np.argmax(belief))


def collect_predictions(beliefs: np.ndarray, labels: dict) -> (list, list):
    """Collect predicted and true labels."""
    num_classes = max(labels.values()) + 1
    y_pred, y_true = [], []
    for node, label in labels.items():
        y_pred.append(predict_label(beliefs, node, num_classes))
        y_true.append(label)
    return y_pred, y_true


def compute_accuracy(beliefs: np.ndarray, labels: dict) -> (float, float):
    """Compute micro and macro F1 scores of prediction."""
    y_pred, y_true = collect_predictions(beliefs, labels)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        return micro_f1, macro_f1


def infer(model: BPModel or JuncTreeModel,
          subgraph: Graph,
          ktree: JunctionTree,
          obs_labels: dict,
          divider: int) -> np.ndarray:
    """Run inference and return the resulting beliefs."""
    if type(model) == BPModel:
        return model.infer(subgraph, obs_labels, divider)
    else:
        return model.infer(subgraph, obs_labels, ktree, divider=1)


def print_result(beliefs: np.ndarray,
                 test_labels: dict,
                 method: str,
                 fold: int,
                 starting_time: float):
    """Print results of inference."""
    micro_f1, macro_f1 = compute_accuracy(beliefs, test_labels)
    print('Accuracy of {}: ({:.4f}, {:.4f})'
          .format(method, micro_f1, macro_f1), end=' ')
    print('in fold {} ({:.2f} seconds).'
          .format(fold + 1, time.time() - starting_time))


@click.command()
@click.argument('edges', default='../../res/graphs/polblogs/edges.csv')
@click.argument('labels', default='../../res/graphs/polblogs/labels.csv')
@click.option('--bound', default=16, type=int)
@click.option('--divider', default=256, type=int)
@click.option('--num-folds', default=3, type=int)
@click.option('--rand-seed', default=0, type=int)
@click.option('--save-dir', default='../../out/subgraphs', type=str)
def main(edges: str, labels: str, bound: int, divider: int, num_folds: int,
         rand_seed: int, save_dir: str):
    """Main function that runs BTW and inference."""

    starting_time = time.time()

    graph = Graph()
    graph.from_csv(edges)

    print('The given graph has {} edges.'.format(graph.num_edges()))

    subgraph, ktree = sample(graph, bound)

    print('A subgraph of {} edges was generated ({:.2f} seconds).'
          .format(subgraph.num_edges(), time.time() - starting_time))

    subgraph.to_csv('{}/subgraph.csv'.format(save_dir))
    ktree.to_csv('{}/ktree.csv'.format(save_dir))
    print("The subgraph and k-tree were saved in '{}'.".format(
        os.path.abspath(save_dir)))

    labels_dict = read_labels(labels)
    chunks = divide_nodes(labels_dict, num_folds, rand_seed)

    for method, model in [('LBP', BPModel()), ('JT', JuncTreeModel())]:
        for fold, chunk in enumerate(chunks):
            obs_labels, test_labels = divide_labels(labels_dict, chunk)
            starting_time = time.time()
            beliefs = infer(model, subgraph, ktree, obs_labels, divider)
            print_result(beliefs, test_labels, method, fold, starting_time)


if __name__ == '__main__':
    main()
