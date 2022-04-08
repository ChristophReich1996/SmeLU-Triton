# Smooth ReLU in Triton

<p align="center">
<img src="plots/SemLU.png" alt="drawing" width="300"/><img src="plots/SemLU_grad.png" alt="drawing" width="300"/>
</p>

**This repository extends my [SmeLU repository](https://github.com/ChristophReich1996/SmeLU) with a [Triton](https://github.com/openai/triton) implementation of the Smooth ReLU activation**

Unofficial **Triton/PyTorch** reimplementation of the Smooth ReLU (SmeLU) activation function proposed in the
paper [Real World Large Scale Recommendation Systems Reproducibility and Smooth Activations](https://arxiv.org/pdf/2202.06499.pdf)
by Gil I. Shamir and Dong Lin.

**This repository includes an easy-to-use pure Triton/PyTorch implementation of the Smooth ReLU.**

## Installation

For using the [Triton](https://github.com/openai/triton) implementation the nightly release of Triton needs to be installed.

````shell script
pip install -U --pre triton
````

If Triton is installed the SmeLU can be installed by using `pip`.

````shell script
pip install git+https://github.com/ChristophReich1996/SmeLU-Triton
````

## Example Usage

The SmeLU can be simply used as a standard `nn.Module`:

````python
import torch
import torch.nn as nn
from smelu import SmeLU

network: nn.Module = nn.Sequential(
    nn.Linear(2, 2),
    SmeLU(),
    nn.Linear(2, 2)
)

network.cuda()

output: torch.Tensor = network(torch.rand(16, 2).cuda())
````

For a more detailed examples on hwo to use this implementation please refer to the [example](example.py) file (requires
Matplotlib to be installed).

The SmeLU takes the following parameters.

| Parameter | Description | Type |
| ------------- | ------------- | ------------- |
| beta | Beta value if the SmeLU activation function. Default 2. | float |

## Runtime Results

| Implementation | Relative speed (forward and backward) in `nn.ReLU()` |
| ------------- | ------------- |
| SmeLU pure PyTorch (on GPU) | ~10 x `nn.ReLU()` | 
| SmeLU Triton/PyTorch (on GPU) | ~1.6 x `nn.ReLU()` | 

Runtime experiments performed on Google Colab.

## Reference

````bibtex
@article{Shamir2022,
        title={{Real World Large Scale Recommendation Systems Reproducibility and Smooth Activations}},
        author={Shamir, Gil I and Lin, Dong},
        journal={{arXiv preprint arXiv:2202.06499}},
        year={2022}
}
````
