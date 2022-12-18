# DART: A Hierarchical Transformer for Document-level Aspect-based Sentiment Classification

<div align="center">
## DART

[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.10-blue)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning_1.7+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg?labelColor=gray)](https://github.com/YanZehong/dart#license)
</div>

This work introduces a hierarchical Transformer-based architecture called **DART** (**D**ocument-level **A**spect-based **R**epresentation from **T**ransformers) which effectively encodes information at three levels, namely token, sentence, and document level. DART employs an attentive aggregation mechanism to learn aspect-specific document representation for sentiment classification.

## Table of Contents
- [Project Structure](#project-structure)
- [Project Set Up](#project-set-up)
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
│   │
│   └── cfg.yaml          <- Main config for training
│
├── dataset            <- Project data
├── datamodules        <- Datamodules (TripAdvisor, BeerAdvocate, PerSenT)
├── models             <- Models (DART, Longformer, BigBird)
├── logs               <- Logs generated by hydra and lightning loggers
├── outputs            <- Save generated data
├── utils                  <- Utility scripts
│
├── run.py                  <- Run Training and evaluation
└── README.md
``` 

## Project Set Up

### 🚀Quickstart
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

You can override any parameter from command line like this
```bash
python run.py train.num_epochs=10 train.batch_size=32
```

### Train on GPU and multi-GPU
```bash
# train on 1 GPU
python run.py gpu=1

# train with DDP (Distributed Data Parallel) (3 GPUs)
python run.py gpu=[0,1,2]

```

> **Warning**: Currently there are problems with DDP mode, read [this issue](https://github.com/ashleve/lightning-hydra-template/issues/393) to learn more.


### Train model with chosen experiment configuration from configs/experiment/

### Test a checkpoint




## FAQ

#### What license is this library released under?

All code *and* models are released under the Apache 2.0 license. See the
`LICENSE` file for more information.

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

For personal communication related to DART, please contact Yan Zehong(`yanzehong1101@outlook.com`). 
<table>
  <tr>
    <td align="center"><a href="https://github.com/YanZehong"><img src="https://github.com/YanZehong.png" width="100px;" alt=""/><br /><sub><b>Yan Zehong</b></sub></a><br /><a href="https://github.com/YanZehong/CS5242-ABSC" title="Code">💻</a></td>
  </tr>
</table>
