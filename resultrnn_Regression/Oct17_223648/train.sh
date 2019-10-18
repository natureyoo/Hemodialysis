#!/usr/bin/env bash

BASH_FILE='./train.sh'
RESULT_DIR='./result'

python3 ./src/run.py \
--save_result_root=$RESULT_DIR \
--bash_file=${BASH_FILE} \
--only_train=False \
--target_type='Regression' \
--model_type='rnn' \
\
--lr=0.001 \
--weight_decay=0.001 \
--max_epoch=10 \
--hidden_size=128 \
--batch_size=16 \
\
--snapshot_epoch_freq=1 \
--valid_iter_freq=1000 \