#! /bin/bash

# Documentation of how to write ELMo vectors to disk for a raw text file.
# $1: config file
# $2: weight file
# $3: input raw text file
# $4: output filepath for vectors to be written

echo "Using ELMo config file at $1"
echo "Using weight file at $2"
echo "Constructing vectors for the whitespace-tokenized sentence-per-line file at $3"
echo "Writing vectors to disk at filepath $4"

allennlp elmo --all --options-file big.options --weight-file elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5  --cuda-device 0 /u/scr/johnhew/data/lstm-word-order/ptb-wsj-sd/raw.dev.txt /u/scr/johnhew/data/lstm-word-order/ptb-wsj-sd/raw.dev.elmo-layers.hdf5
