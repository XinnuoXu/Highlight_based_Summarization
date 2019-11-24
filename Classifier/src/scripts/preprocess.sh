#!/bin/bash

JSON_PATH=/scratch/xxu/bert/jsons.cls/
BERT_DATA_PATH=/scratch/xxu/bert/data.cls/

python preprocess.py \
	-mode format_to_bert \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
      	-lower \
	-n_cpus 10 \
	-log_file ../logs/preprocess.log
