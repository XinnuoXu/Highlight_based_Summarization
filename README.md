# Highlight_based_Summarization

## Environment setup

### Step1: Install pytorch env

```
conda create -n Highlight python=3.6
conda activate Highlight
conda install pytorch=1.1.0 torchvision cudatoolkit=10.0 -c pytorch
(or conda install pytorch torchvision cudatoolkit=10.0 -c pytorch; conda install pytorch=1.1.0 -c soumith)

pip install pytorch_transformers
pip install pyrouge
pip install tensorboardX
```

### Step2: Install allennlp

```
pip install allennlp
wget https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz
```

## System attacking with 50 documents
```
cd Fact_extraction
```
Each document has a document ID, for example 36588641. For system attacking, you can write down your summaries in file `debug/doc_ID.data`, for example `debug/36588641.data`. Then run this line to get their semantic role labelling results:
```
python fact_extraction_for_debug.py srl 36588641
```
and this line to get the fact tree structure:
```
python fact_extraction_for_debug.py tree 36588641
```
It will generate a directory `debug/doc_ID.tree/`, in this case `debug/36588641.tree/`.
