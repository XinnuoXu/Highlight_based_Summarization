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
### Step1: Create your examples
To try some DIY examples, you should write your examples down in `input.src` and `input.tgt`. `input.src` is for document trees and `input.tgt` is for summaries(just sentences, not trees). Note that

* the number of lines in `input.src` is equal to `input.tgt`
* lines in `input.src` should be the copy of the first row (sorry for the duplication. I will make it decent soon)
* the first row in `input.tgt` should be the ground truth. Please find some examples in `input.src` and `input.tgt` in this repository. 

### Step2: Get Corr scores and debug information

Run `sh run.sh`. it will create a directory called `output/` containing following files:

* `corr.txt` is the corr scores for each examples. It's one row per example with the format `document \t summary \t Corr_F \t Corr_A` for each row.
* `{k}.txt` are details for fact/argument distances in the kth example (<img src="http://latex.codecogs.com/gif.latex?d_{ij}^f" border="0"/> and <img src="http://latex.codecogs.com/gif.latex?d_{ij}^a" border="0"/> in the paper). 
* `{k}.hl` are fact level weights <img src="http://latex.codecogs.com/gif.latex?\mathbf{w}_\ast^f" border="0"/> for the kth example.

### Step3: Visualization

To visualize the fact/argument distances and fact level weights, you can copy the file you are interested in to `../display/` and replace it to the file `attn_vis_data.json`. Then run

```
sh run_service.sh
```

You can view the visualization result by visiting http://localhost:8000 is your web browser.

![alt text](https://github.com/XinnuoXu/Highlight_based_Summarization/blob/master/display/distance.png)



