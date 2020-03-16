# CSE517 Final Project

We attempt to reproduce results from the paper EMNLP 2019 paper [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings](<[https://arxiv.org/abs/1909.00512](https://arxiv.org/abs/1909.00512)>). We also extend the analysis performed in the paper to 3 new models released after the paper was published: XLNet, XLM, and RoBERTa.

## Setup

We recommend setting up a virtual environment for this project. For example, [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

Install requirements

```
pip install -r contextual/requirements.txt
```

## Data

Fetch the datasets from the SemEval STS tasks 2012-2016 with

```Shell
cd dataset
chmod +x fetch_data.sh
./fetch_data.sh
```

Create a file named sts.csv with all the sentences together.

```Python
python make_dataset.py
```

## Analysis

Ensure that your dataset is stored in a file named `sts.csv` in the `contextual` directory.

Fetch embeddings from all layers for all models

```Python
python preprocess.py
python new_preprocess.py
```

Perform analysis

```Python
python analyze.py
```

Visualize the results

```Python
python visualize.py
```
