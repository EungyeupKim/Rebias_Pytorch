# Rebias_Pytorch

Pytorch implemenatation of Rebias (Bahng et al., Learning De-Biased Representations with Biased Representations, ICML'20)

## Getting Started

The repository contains the implementation of Rebias on biased MNIST-CIFAR10 dataset (MNIST digit on the foreground and CIFAR-10 on the background).
This dataset represents the correlation within a pair of MNIST digit class and CIFAR-10 class, which is defined as a bias in this project.
Morever, we provide classification accuracy of Rebias, Vanilla and Biased model and in-depth analysis on them. 
Specifically, it provides 
* Generation of MNIST-CIFAR10 dataset with various degrees of correlations
* Train and evaluation of Rebias and its baseline models on MNIST-CIFAR10 dataset
* Comparisons and analysis on the classification accuracies of Rebias and its baseline models

Note that, as original paper recommended, AdamP (https://arxiv.org/abs/2006.08217) optimizer is utilized in this implementation.

## Installation

### Dependencies
Install dependencies with following command.
```
pip install -r requirements.txt
```

### Train and Test Datasets
Generate Biased MNIST-CIFAR10 datasets with varying correlation (1.0, 0.999, 0.997, 0.995, 0.99, 0.9, 0.1)
```
python dataset.py --rhos 1.0 0.999 0.997 0.995 0.990 0.90 0.1
```
Above command line generates both train and test sets for each correlation, saving under the default dataset path.
Note that we use the two evaluation dataset, biased and unbiased sets, as in the original paper.
* Biased dataset: Same correlation as the training datasets.
* Unbiased dataset: Set the correlation 0.1 for uniform distribution sampling.

### How to Run
Train models on the MNIST-CIFAR10 dataset with given correlation.
```
python main.py --name default --root_dir /path/to/the/dataset/ --rho 0.999
python main.py --name default --root_dir /path/to/the/dataset/ --rho 0.997
python main.py --name default --root_dir /path/to/the/dataset/ --rho 0.995
python main.py --name default --root_dir /path/to/the/dataset/ --rho 0.99
python main.py --name default --root_dir /path/to/the/dataset/ --rho 0.9
```
This command line trains 3 models, Rebias and its baseline models (Vanilla and Biased), simultaneously.

Test models on biased (given correlation) and unbiased datasets.
```
python evaluation.py --name default --root_dir /path/to/the/dataset/ --load_dir /path/to/the/checkpoint/ --load_epoch 80 --rho 0.999
python evaluation.py --name default --root_dir /path/to/the/dataset/ --load_dir /path/to/the/checkpoint/ --load_epoch 80 --rho 0.997
python evaluation.py --name default --root_dir /path/to/the/dataset/ --load_dir /path/to/the/checkpoint/ --load_epoch 80 --rho 0.995
python evaluation.py --name default --root_dir /path/to/the/dataset/ --load_dir /path/to/the/checkpoint/ --load_epoch 80 --rho 0.99
python evaluation.py --name default --root_dir /path/to/the/dataset/ --load_dir /path/to/the/checkpoint/ --load_epoch 80 --rho 0.9
```
This command line results in Top-1 accuracies of 3 models, Rebias and its baseline models (Vanilla and Biased), simultaneously.

## Note
This implementation partially refers to the official Pytorch implementation of Rebias (https://github.com/clovaai/rebias).

Referred points are mentioned below:

* Python file ```models.py``` is taken from ```./models/mnist_models.py``` in official implemenation repository.
* Implementation of Hilbert-Schmidt Independence Criterion is partially referred from ```./criterions/hsic.py``` in official implemenation repository.

All the referred points are indicated in the codes in this repository.
