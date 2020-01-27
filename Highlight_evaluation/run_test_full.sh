BASE_PATH="../Fact_extraction/doc_predictions/"
OUTPUT_PATH="/scratch/xxu/system_trees/"
export CUDA_VISIBLE_DEVICES=0
# ptgen full
#nohup python get_bert_highlight.py highlight_simple_format $BASE_PATH/ptgen_pred.src $BASE_PATH/ptgen_gold.tree $OUTPUT_PATH/ptgen_gold.alg &
#nohup python get_bert_highlight.py highlight_simple_format $BASE_PATH/ptgen_pred.src $BASE_PATH/ptgen_tgt.tree $OUTPUT_PATH/ptgen_full.alg & 

export CUDA_VISIBLE_DEVICES=2
# tconvs2s full
#cp $OUTPUT_PATH/ptgen_gold.alg $OUTPUT_PATH/tconvs2s_gold.alg
#nohup python get_bert_highlight.py highlight_simple_format $BASE_PATH/tconvs2s_pred.src $BASE_PATH/tconvs2s_tgt.tree $OUTPUT_PATH/tconvs2s_full.alg &

export CUDA_VISIBLE_DEVICES=2
# bertalg
#nohup python get_bert_highlight.py highlight_simple_format $BASE_PATH/bertalg_pred.src $BASE_PATH/bertalg_gold.tree $OUTPUT_PATH/bertalg_gold.alg &
#nohup python get_bert_highlight.py highlight_simple_format $BASE_PATH/bertalg_pred.src $BASE_PATH/bertalg_tgt.tree $OUTPUT_PATH/bertalg_full.alg &

# bert full
#nohup python get_bert_highlight.py highlight_simple_format $BASE_PATH/bert_pred.src $BASE_PATH/bert_gold.tree $OUTPUT_PATH/bert_gold.alg &
#nohup python get_bert_highlight.py highlight_simple_format $BASE_PATH/bert_pred.src $BASE_PATH/bert_tgt.tree $OUTPUT_PATH/bert_full.alg & 

echo "****SYSTEM****"
echo "BERT_ALG"
python alignment_ROUGE.py auto_full bertalg
echo "PT"
python alignment_ROUGE.py auto_full ptgen
echo "TCONV"
python alignment_ROUGE.py auto_full tconvs2s
echo "BERT"
python alignment_ROUGE.py auto_full bert
