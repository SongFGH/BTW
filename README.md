# Bounded Treewidth (BTW) Sampling

This is a code repository for "Efficient Sampling of Subgraphs with Bounded
Treewidth for Fast and Accurate Graphical Inference," submitted to KDD 2019.

## 1. Abstract

How can we run graphical inference on a large graph efficiently and accurately?
Many real-world networks are modeled as graphical models, and graphical
inference is fundamental to understand the properties of those networks. In this
work, we propose a novel approach for fast and accurate inference, which  first
samples a small subgraph and then runs inference over the subgraph instead of
the given graph. This is done by the bounded treewidth (BTW) sampling, our novel
algorithm that generates a subgraph with bounded treewidth while retaining as
many edges as possible. We first analyze the properties of BTW theoretically.
Then, we evaluate our approach on node classification and compare it with the
baseline which is to run loopy belief propagation (LBP) on the original graph.
Coupled with the junction tree algorithm, our approach is more accurate up to
13.7%. Coupled with LBP, our approach shows comparable accuracy with faster
computational time up to 23.4 times. We further compare BTW with other graph
sampling algorithms and show that it consistently outperforms all competitors.

## 2. Code Information

This repository contains the source code of BTW and inference algorithms that 
were used in the experiments of the paper. Given an undirected network, BTW 
generates a subgraph with bounded treewidth. The result will slightly change
every time you run the code since it contains randomness. All codes are
written by Python 3.5.

The structure of this repository is given as follows:
- `src` contains the source code of BTW and inference algorithms.
- `res` contains public graph datasets.
- `demo.sh` is a demo script that runs the code with pre-defined parameters.
- `Makefile` makes it easy to run the demo script by just typing 'make all'.
- `requirements.txt` contains required Python packages.

Four public datasets are included in `res`. Each dataset was preprocessed and 
consists of two files: `edges.csv` and `labels.csv`. The former contains the
undirected edges as a CSV format, and the latter contains the labels of observed 
nodes. The labels represent important properties of the nodes such as research 
areas of articles and political orientations of web blogs, and are needed only
for inference, not BTW. 

## 3. How to Run

You may type `make all` to run the demo script. It runs BTW for the 'PolBlogs'
dataset and inference by both LBP and the junction tree algorithm. Otherwise,
you can run the main code directly by providing parameters that you want.

```
- Usage: python -m btw [OPTIONS] [EDGES] [LABELS]

- Arguments:
  EDGES                Path of an edge file.
  LABELS               Path of a label file.

- Options:
  --bound INTEGER      Treewidth bound (k) for BTW.
  --divider INTEGER    Skewness parameter (d) for convergence of LBP.
  --num-folds INTEGER  Number of folds for inference.
  --rand-seed INTEGER  Random seed for dividing the nodes.
  --save-dir STRING    Path to save a subgraph generated from BTW.
```

If you run the script, BTW generates a subgraph with bounded treewidth, and LBP
and the junction tree algorithm are run on the generated subgraph. The micro and 
macro F1 scores are printed on a standard output as well as the inference time.
