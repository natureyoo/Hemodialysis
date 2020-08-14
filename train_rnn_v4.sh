#!/usr/bin/env bash

BASH_FILE='./train_rnn_v4.sh'
RESULT_DIR='result'
RM_version='no_remove'
# LOAD_PATH='./trained_models/4epoch.model'

python3 -W ignore ./src/run_rnn_v4_0813.py \
--save_result_root=$RESULT_DIR \
--bash_file=${BASH_FILE} \
--model_type='rnn_v4' \
--optim='Adam' \
\
--input_fix_size=83 \
--input_seq_size=43 \
\
--lr=5e-4 \
--lr_decay_rate=0.7 \
--weight_decay=5e-4 \
--hidden_size=64 \
--rnn_hidden_layers 3 \
--dropout_rate 0.4  \
--batch_size=512 \
\
--weight_loss_ratio 10.0 \
--topk_loss_ratio 0.0 \
\
--init_epoch=0 \
--remove_version=$RM_version \
# --load_path=$LOAD_PATH

