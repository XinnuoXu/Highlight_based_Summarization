#!/bin/bash

RAW_PATH=../../Document_highlight.BERT/highlights.bert/s2s_alignment/xsum_
JSON_PATH=../test_json/xsum

python preprocess.py \
	-mode format_xsum_shard_only \
	-raw_path ${RAW_PATH} \
	-save_path ${JSON_PATH} \
	-n_cpus 10 \
	-log_file ../logs/cnndm.log \
