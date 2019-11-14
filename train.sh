#!/usr/bin/env bash

BASH_FILE='./train.sh'
RESULT_DIR='result'

python3 -W ignore ./src/run_rnn.py \
--save_result_root=$RESULT_DIR \
--bash_file=${BASH_FILE} \
--only_train=False \
--target_type='Classification' \
--model_type='rnn' \
--optim='SGD' \
\
--lr=1e-2 \
--lr_decay_rate=0.1 \
--weight_decay=0.00000 \
--max_epoch=1000 \
--hidden_size=256 \
--rnn_hidden_layers 2 \
--dropout_rate 0.0 \
--batch_size=32 \
\
--snapshot_epoch_freq=1 \
--valid_iter_freq=1000 \
--train_print_freq=50 \
