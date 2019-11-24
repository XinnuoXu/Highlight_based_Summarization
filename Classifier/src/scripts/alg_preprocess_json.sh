#!/bin/bash

RAW_PATH=/scratch/xxu/bert/s2s_alignment/xsum_
JSON_PATH=/scratch/xxu/bert/jsons/xsum

python preprocess.py \
	-mode format_xsum_shard_only \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-n_cpus 10 \
	-log_file ../logs/cnndm.log \
