#!/bin/bash

BERT_DATA_PATH=/scratch/xxu/bert/data.cls/xsum
MODEL_PATH=/scratch/xxu/bert/models.cls/

python train.py  \
	-task ext \
	-mode train \
	-bert_data_path ${BERT_DATA_PATH} \
	-ext_dropout 0.1  \
	-model_path ${MODEL_PATH} \
	-lr 2e-3 \
	-save_checkpoint_steps 1000 \
	-batch_size 300 \
	-train_steps 60000 \
	-report_every 50 \
	-accum_count 2 \
	-use_interval true \
	-warmup_steps_bert 20000 \
	-warmup_steps_dec 10000 \
	-visible_gpus 0,1,2  \
	-max_pos 512 \
	-log_file ../logs/ext_bert_cnndm
