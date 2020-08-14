#!/usr/bin/env bash

BASH_FILE='./train_rnn_v3.sh'
RESULT_DIR='result'
RM_version='no_remove'
# LOAD_PATH='./trained_models/4epoch.model'

python3 -W ignore ./src/run_rnn_v3.py \
--save_result_root=$RESULT_DIR \
--bash_file=${BASH_FILE} \
--only_train=False \
--target_type='Classification' \
--model_type='rnn_v3' \
--optim='Adam' \
\
--input_fix_size=110 \
--input_seq_size=8 \
\
--lr=2e-4 \
--lr_decay_rate=0.7 \
--weight_decay=5e-5 \
--max_epoch=30 \
--hidden_size=1024 \
--rnn_hidden_layers 4 \
--dropout_rate 0.3 \
--batch_size=2048 \
\
--snapshot_epoch_freq=1 \
--valid_iter_freq=5000 \
--train_print_freq=50 \
--init_epoch=0 \
--remove_version=$RM_version \
# --load_path=$LOAD_PATH

