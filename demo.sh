#!/usr/bin/env bash

cd src
python -m btw \
    ../res/graphs/polblogs/edges.csv \
    ../res/graphs/polblogs/labels.csv \
    --bound 8 \
    --divider 256 \
    --num-folds 3 \
    --rand-seed 0 \
    --save-dir ../out
