#python get_bert_highlight.py highlight
#python get_bert_highlight.py highlight system_trees/system_ptgen.tree/
#python get_bert_highlight.py highlight system_trees/system_bert.tree/
#python get_bert_highlight.py highlight system_trees/system_bertalg.tree/
#python get_bert_highlight.py highlight system_trees/system_tconvs2s.tree/

echo "****CORR fact****"
python alignment_check.py fact
echo "****CORR phrase****"
python alignment_check.py phrase
echo "****HUMAN****"
echo "BERT"
python alignment_ROUGE.py human system_bert.alg
echo "BERT_ALG"
python alignment_ROUGE.py human system_bertalg.alg
echo "PT"
python alignment_ROUGE.py human system_ptgen.alg
echo "TCONV"
python alignment_ROUGE.py human system_tconvs2s.alg
echo "****SYSTEM****"
echo "BERT"
python alignment_ROUGE.py system system_bert.alg
echo "BERT_ALG"
python alignment_ROUGE.py system system_bertalg.alg
echo "PT"
python alignment_ROUGE.py system system_ptgen.alg
echo "TCONV"
python alignment_ROUGE.py system system_tconvs2s.alg
