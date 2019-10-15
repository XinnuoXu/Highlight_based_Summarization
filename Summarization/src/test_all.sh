#!/bin/bash

BERT_DATA_PATH=/scratch/xxu/bert/xsum
MODEL_PATH=/scratch/xxu/bert/models/

python train.py \
	-task abs \
	-mode validate \
	-test_all \
	-batch_size 3000 \
	-test_batch_size 500 \
	-bert_data_path ${BERT_DATA_PATH} \
	-log_file ../logs/val_abs_bert_cnndm \
	-model_path ${MODEL_PATH} \
	-sep_optim true \
	-use_interval true \
	-visible_gpus 2 \
	-max_pos 512 \
	-max_length 200 \
	-alpha 0.95 \
	-min_length 50 \
	-result_path ../logs/abs_bert_cnndm
