#! /usr/bin/env #!/usr/bin/env bash

BASH_FILE="./train_mlp.sh"
RESULT_DIR="./result/"

python ./src/run.py \
--only_train=False \
--model_type='mlp' \
--description='Regression with bn' \
\
--lr=1e-2 \
--max_epoch=10000 \
\
--target_type='Regression' \
--batch_size=64 \
--hidden_size=512 \
--snapshot_epoch_freq=5 \
--valid_iter_freq=5000 \
--weight_decay=0.0000 \
--sampler=0 \
--weighted=1 \
\
--save_result_root=$RESULT_DIR \
--bash_file=${BASH_FILE} \
# --resume \
