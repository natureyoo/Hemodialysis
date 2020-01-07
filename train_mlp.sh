#! /usr/bin/env #!/usr/bin/env bash

BASH_FILE="./train_mlp.sh"
RESULT_DIR="./result/"

python ./src/run_mlp.py \
--save_result_root=$RESULT_DIR \
--bash_file=${BASH_FILE} \
\
--only_train=False \
--model_type='mlp' \
--target_type='cls' \
--optim='SGD' \
\
--lr=0.01 \
--lr_decay_rate=0.1 \
--weight_decay=5e-6 \
--max_epoch=30 \
--hidden_size=256 \
--dropout_rate 0.1 \
--batch_size=32 \
\
--snapshot_epoch_freq=1 \
--valid_iter_freq=5000 \
--train_print_freq=50 \
--init_epoch=0 \
# --resume \
