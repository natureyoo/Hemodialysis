#!/usr/bin/env bash


###############################################################
#########################  111111  ############################
BASH_FILE='./MLP.sh'
# LOAD_PATH='./result/rnn_v4/Classification/20200817_115658/bs1024_lr0.0002_wdecay5e-05/snapshot/7epoch.model'

python3 -W ignore ./src/run_MLP.py \
--bash_file=${BASH_FILE} \
--optim='Adam' \
\
--num_feature=260 \
\
--lr=5e-4 \
--lr_decay_rate=0.7 \
--weight_decay=5e-5 \
--hidden_size=1024 \
--hidden_layers 4 \
--dropout_rate 0.3  \
--batch_size=1024 \
\
# --load_path=$LOAD_PATH