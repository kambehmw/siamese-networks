#!/usr/bin/env bash

# dataset repository https://github.com/brendenlake/omniglot
IMAGES_BACKGROUND=https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
IMAGES_EVALUATION=https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip

DATASET_PATH=data/omniglot

echo "make data directory."
mkdir -p $DATASET_PATH

echo "dataset downloading and unpacking..."
wget -P $DATASET_PATH $IMAGES_BACKGROUND
unzip $DATASET_PATH/images_background.zip -d $DATASET_PATH
wget -P $DATASET_PATH $IMAGES_EVALUATION
unzip $DATASET_PATH/images_evaluation.zip -d $DATASET_PATH

echo "dataset downloading and unpacking completed!"