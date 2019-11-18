#!/bin/bash

JSON_PATH=../test_json/
BERT_DATA_PATH=../test_data/

python preprocess.py \
	-mode format_to_bert \
	-raw_path ${JSON_PATH} \
	-save_path ${BERT_DATA_PATH} \
      	-lower \
	-n_cpus 4 \
	-log_file ../logs/preprocess.log
