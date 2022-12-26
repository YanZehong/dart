# DART: A Hierarchical Transformer for Document-level Aspect-based Sentiment Classification

<div align="center">

[![Python Versions](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.12+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning_1.7+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg?labelColor=gray)](https://github.com/YanZehong/dart#license)
</div>

This work introduces a hierarchical Transformer-based architecture called **DART** (**D**ocument-level **A**spect-based **R**epresentation from **T**ransformers) which effectively encodes information at three levels, namely token, sentence, and document level. DART employs an attentive aggregation mechanism to learn aspect-specific document representation for sentiment classification.

## Table of Contents
- [Project Structure](#project-structure)
- [Quickstart](#quickstart-)
  * [Installation](#installation)
  * [Fine-tuning with DART](#fine-tuning-with-dart)
- [Introduction](#introduction)
  * [DART Architecture](#dart-architecture)
  * [Dataset](#dataset)
- [FAQ](#faq)
- [Citation](#citation)

## Project Structure
The directory structure of this project is:
```
├── configs            <- Hydra configuration files
│   ├── logdir            <- Logger configs
│   ├── data              <- Datamodule configs
│   ├── model             <- Modelmodule configs
│   ├── experiment        <- Experiment configs
│   └── cfg.yaml          <- Main config for training
│
├── dataset            <- Project data
├── datamodules        <- Datamodules (TripAdvisor, BeerAdvocate, PerSenT)
├── models             <- Models (DART, Longformer, BigBird)
├── logs               <- Logs generated by hydra and lightning loggers
├── outputs            <- Save generated data
├── utils              <- Utility scripts
│
├── run.py             <- Run Training and evaluation
└── README.md
``` 

## Quickstart 🚀

### Installation
Install dependencies.

```
# clone project
git clone https://github.com/YanZehong/dart
cd dart

# [OPTIONAL] create conda environment
conda create -n myenv -y python=3.10 pip
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/
# conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt
```

> **Note**: To install requirements, run `pip install -r requirements.txt`. Please ensure that you have met the prerequisites in [PyTorch](https://pytorch.org/) and install correspond version. 


### Fine-tuning with DART
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```bash
python run.py gpu=1 experiment=tripadvisor_dart.yaml
```

Train model with default configuration
```bash
# train on 1 GPU
python run.py gpu=1

# train with DDP (Distributed Data Parallel) (3 GPUs)
python run.py gpu=[0,1,2]
```
> **Warning**: Currently there are problems with DDP mode, read [this issue](https://github.com/Lightning-AI/lightning/issues/8375) to learn more.

You can override any parameter from command line like this
```bash
python run.py gpu=3 train.num_epochs=10 train.batch_size=32
```

<details>
<summary><b>Use Miniconda for GPU environments</b></summary>

Use miniconda for your python environments (it's usually unnecessary to install full anaconda environment, miniconda should be enough).
It makes it easier to install some dependencies, like cudatoolkit for GPU support. It also allows you to access your environments globally.

Example installation:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Create new conda environment:

```bash
conda create -n myenv python=3.10
conda activate myenv
```

</details>

<details>
<summary><b>Use torchmetrics</b></summary>

Use official [torchmetrics](https://github.com/PytorchLightning/metrics) library to ensure proper calculation of metrics. This is especially important for multi-GPU training!

For example, instead of calculating accuracy by yourself, you should use the provided `Accuracy` class like this:

```python
from torchmetrics import Accuracy

class ModelModel(LightningModule):
    def __init__(self)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        ...
        acc = self.train_acc(predictions, targets)
        self.log("train/acc", acc)
        ...

    def validation_step(self, batch, batch_idx):
        ...
        acc = self.val_acc(predictions, targets)
        self.log("val/acc", acc)
        ...
```

Make sure to use different metric instance for each step to ensure proper value reduction over all GPU processes.

Torchmetrics provides metrics for most use cases, like F1 score or confusion matrix. Read [documentation](https://torchmetrics.readthedocs.io/en/latest/#more-reading) for more.

</details>

<details>
<summary><b>Follow PyTorch Lightning style guide</b></summary>

The style guide is available [here](https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html).<br>
</details>

## Introduction

### DART Architecture

DART overcomes the restriction of 512 tokens by splitting the long document into sentences or chunks of less than 512 tokens, and processing each sentence/chunk before aggregating the results.
Figure \ref{fig:dart} shows the proposed DART framework  that takes as input a document $d$ and an aspect $a_j$ and output the document representation $\hat{d}_j$ with respect to $a_j$.
There are three key blocks in DART:

- Sentence Encoding Block.  
- Global Context Interaction Block.  
- Aspect Aggregation Block.  

### Dataset

|      Dataset      |      \#aspects      |      \#docs      | \#long docs (\%)| \#sentences/doc |  \#tokens/doc | \#tokens/sentence
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| TripAdvisor | 7 | 28543 | 4027 (14.1\%) | 12.9 | 298.9 | 23.1|
| BeerAdvocate| 4 | 27583 | 217 (0.8\%) | 11.1 | 173.5 | 15.7|
|PerSenT | 6 | 4512 | 1031 (22.9\%) | 17.5 | 389.8 | 22.2  |


## FAQ

#### What license is this library released under?

All code *and* models are released under the Apache 2.0 license. See the
`LICENSE` file for more information.

#### I am getting out-of-memory errors, what is wrong?
All experiments in the paper were fine-tuned on a GPU/GPUs, which has 40GB of device RAM. Therefore, when using a GPU with 12GB - 16GB of RAM, you are likely to encounter out-of-memory issues if you use the same hyperparameters described in the paper. Additionally, different models require different amount of memory. Available memory also depends on the accelerator configuration (both type and count).

The factors that affect memory usage are:  
-  **`data.max_num_seq`**: You can fine-tune with a shorter max sequence length to save
    substantial memory. 

-   **`train.batch_size`**: The memory usage is also directly proportional to
    the batch size. You could decrease the `train.batch_size=8` (and decrease `train.lr`
    accordingly) if you encounter an out-of-memory error.

-   **Model backbone type, `base` vs. `large`**: The `large` model
    requires significantly more memory than `base`.


## Citation

If you find this work useful, please cite as following:

```
@article{2023dart,
  title={DART: A Hierarchical Transformer for Document-level Aspect-based Sentiment Classification},
  author={},
  journal={},
  year={}
}
```

If we submit the paper to a conference or journal, we will update the BibTeX.


## Contact information

For help or issues using DART, please submit a GitHub issue.

For personal communication related to DART, please contact me. 
<table>
  <tr>
    <td align="center"><a href="https://github.com/YanZehong"><img src="https://github.com/YanZehong.png" width="100px;" alt=""/><br /><sub><b>Yan Zehong</b></sub></a><br /><a href="https://github.com/YanZehong/dart" title="Code">💻</a></td>
  </tr>
</table>
