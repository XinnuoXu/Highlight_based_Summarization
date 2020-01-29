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
cd Debug
```
To try some DIY examples, you should write your examples down in `input.src` and `input.tgt`. `input.src` is for document trees and `input.tgt` is for summaries(just sentences, not trees). Note that, (1) the number of lines in `input.src` is equal to `input.tgt`; (2) lines in `input.src` should be the copy of the first row (sorry for the duplication. I will make it decent soon); (3) the first row in `input.tgt` should be the ground truth. Please find some examples in `input.src` and `input.tgt` in this repository. 

Then run 
```
sh run.sh
```
it will create a directory called `output/` containing following files:

* `corr.txt` the corr scores for each examples with the format `document \t summary \t Corr_F \t Corr_A`
* `other *.txt` are details for fact/argument distances. <img src="http://latex.codecogs.com/gif.latex?d_{ij}^f" border="0"/> and <img src="http://latex.codecogs.com/gif.latex?d_{ij}^a" border="0"/> in paper. 
* `*.hl` are fact level weights <img src="http://latex.codecogs.com/gif.latex?\mathbf{w}_\ast^f" border="0"/>.






