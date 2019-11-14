#!/usr/bin/env bash

BASH_FILE='./train_rnn_v3.sh'
RESULT_DIR='result'

python3 -W ignore ./src/run_rnn_v3.py \
--save_result_root=$RESULT_DIR \
--bash_file=${BASH_FILE} \
--only_train=False \
--target_type='Classification' \
--model_type='rnn_v3' \
--optim='SGD' \
\
--lr=1e-3 \
--lr_decay_rate=0.1 \
--weight_decay=5e-6 \
--max_epoch=1000 \
--hidden_size=1024 \
--rnn_hidden_layers 3 \
--dropout_rate 0.1 \
--batch_size=16 \
\
--snapshot_epoch_freq=1 \
--valid_iter_freq=1000 \
--train_print_freq=50 \
--init_epoch=0 \