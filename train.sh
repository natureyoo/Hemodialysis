#!/usr/bin/env bash

BASH_FILE='./train.sh'
RESULT_DIR='./result'

python3 ./src/run_v2.py \
--save_result_root=$RESULT_DIR \
--bash_file=${BASH_FILE} \
--only_train=True \
--target_type='Classification' \
--model_type='mlp' \
--optim='SGD' \
--merge_Flag=True \
\
--lr=1e-2 \
--lr_decay_rate=0.1 \
--weight_decay=0.0000 \
--max_epoch=1000 \
--hidden_size=1024 \
--batch_size=64 \
\
--snapshot_epoch_freq=1 \
--valid_iter_freq=8764 \