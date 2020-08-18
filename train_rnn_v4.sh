#!/usr/bin/env bash

BASH_FILE='./train_rnn_v4.sh'
RESULT_DIR='result'
data_root='../data/raw_data/0813/pt_file_58/wo_EF/'
# LOAD_PATH='./result/rnn_v4/Classification/20200817_115658/bs1024_lr0.0002_wdecay5e-05/snapshot/7epoch.model'

python3 -W ignore ./src/run_rnn_v4_0813.py \
--data_root=$data_root \
--save_result_root=$RESULT_DIR \
--bash_file=${BASH_FILE} \
--model_type='rnn_v4' \
--optim='Adam' \
\
--input_fix_size=85 \
--input_seq_size=45 \
\
--lr=5e-4 \
--lr_decay_rate=0.7 \
--weight_decay=5e-5 \
--hidden_size=1024 \
--rnn_hidden_layers 4 \
--dropout_rate 0.3  \
--batch_size=1024 \
\
--weight_loss_ratio 0.0 \
--topk_loss_ratio 1.0 \
\
# --load_path=$LOAD_PATH