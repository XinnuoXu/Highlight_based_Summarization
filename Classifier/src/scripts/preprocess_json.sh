#!/bin/bash

RAW_PATH=../raw_data/xsum_
JSON_PATH=/scratch/xxu/bert/jsons.cls/xsum

python preprocess.py \
	-mode format_xsum_to_lines_easy \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-n_cpus 1 \
	-log_file ../logs/cnndm.log \
