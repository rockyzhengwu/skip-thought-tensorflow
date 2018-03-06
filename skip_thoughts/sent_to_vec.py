#!/usr/bin/env python
#-*-coding:utf-8-*-



# Imports.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

VOCAB_FILE = "../data/vocab.txt"
EMBEDDING_MATRIX_FILE = "../data/exp_vocab/embeddings.npy"

CHECKPOINT_PATH = "../data/model/model.ckpt-1957"

# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.

MR_DATA_DIR = "../data/train.txt"

# Set up the encoder. Here we are using a single unidirectional model.
# To use a bidirectional model as well, call load_model() again with
# configuration.model_config(bidirectional_encoder=True) and paths to the
# bidirectional model's files. The encoder will use the concatenation of
# all loaded models.

encoder = encoder_manager.EncoderManager()

encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

# Load the movie review dataset.
data = []
with open(MR_DATA_DIR, 'r')as f:
    data.extend([line.strip() for line in f if line.strip()])
data = data[:100]
# Generate Skip-Thought Vectors for each sentence in the dataset.
encodings = encoder.encode(data)

# Define a helper function to generate nearest neighbors.
def get_nn(ind, num=10):
  encoding = encodings[ind]
  scores = sd.cdist([encoding], encodings, "cosine")[0]
  sorted_ids = np.argsort(scores)

  print("Sentence:")
  print("", data[ind])
  print("\nNearest neighbors:")
  for i in range(1, num + 1):
    print(" %d. %s (%.3f)" %
          (i, data[sorted_ids[i]], scores[sorted_ids[i]]))

# Compute nearest neighbors of the first sentence in the dataset.
for i in range(10):
    get_nn(i)
