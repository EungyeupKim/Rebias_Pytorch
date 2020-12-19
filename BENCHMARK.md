# Benchmark

Quantitative comparison (Top-1 Accuracy) between Rebias and its baselines and its Analysis

## Experimental Setup
### Models
* Rebias, Vanilla, Biased

### Training
* Train Dataset: MNIST-CIFAR10 dataset with rho = 0.999, 0.997, 0.995, 0.99, 0.9 ```--rho 0.999```
* Opimizer: AdamP ```AdamP(model_f.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)```
* Learning Rate: ```--lr 0.001```
* Learning Rate Scheduling: ```lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)```
* HSIC kernel type: ```--kernel_type 1```

### Evaluation
* Test Dataset: Biased Set (same rho with train set), Unbiased Set (rho = 0.1)
* Top-1 Accuracy
* HSIC Learning Curve

### Additional Experiments on Colored-MNIST dataset (taken from the original implementation repository)
* Train Dataset: Colored-MNIST dataset with rho = 0.999, 0.997, 0.995, 0.99, 0.9 ```--rho 0.999```
* Test Dataset: Biased Set (same rho with train set), Unbiased Set (rho = 0.1)
* Top-1 Accuracy
* HSIC Learning Curve

## Quantitative Results
### Top1 Accuracy on MNIST-CIFAR10 and Colored-MNIST


### Learning Curve of Test Accuracy


### Learning Curve of HSIC



### Analysis


## Note
Most of the implementation details are referred from the original Rebias paper.
