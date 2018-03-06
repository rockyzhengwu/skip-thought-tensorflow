#!/usr/bin/env bash

train_file=../../data/train.txt
output_dir=../../data/tfrecord/

python preprocess_dataset.py --input_file "$train_file" --output_dir "$output_dir"