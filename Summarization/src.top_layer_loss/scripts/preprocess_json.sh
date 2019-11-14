#!/bin/bash

RAW_PATH=../../../fact_data/s2s_bert/xsum_
JSON_PATH=/scratch/xxu/bert/jsons/xsum

python preprocess.py \
	-mode format_xsum_to_lines_easy \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-n_cpus 1 \
	-log_file ../logs/cnndm.log \
