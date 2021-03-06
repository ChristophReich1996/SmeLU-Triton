from typing import Tuple

import torch
from torch import autograd
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _smelu_kernel_forward(
        input_pointer,
        beta: float,
        output_pointer,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
):
    """ Triton kernel SmeLU forward """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(input_pointer + offsets, mask=mask)
    output = tl.where(x >= beta, x, 0.)
    output = tl.where(tl.abs(x) <= beta, ((x + beta) * (x + beta)) / (4. * beta), output)
    # Write-back output
    tl.store(output_pointer + offsets, output, mask=mask)


def _smelu_triton_forward(
        input: torch.Tensor,
        beta: float = 2.
) -> torch.Tensor:
    """
    Wrapper function for SmeLU forward triton kernel
    :param input (torch.Tensor): Input tensor of any shape
    :param beta (float): Beta value of SmeLU
    :return (torch.Tensor): Activation of SmeLU
    """
    # Init output tensor
    output: torch.Tensor = torch.empty_like(input)
    # Make input contiguous if needed
    if not input.is_contiguous():
        input = input.contiguous()
    # Get number of elements in input
    number_of_elements: int = input.numel()
    # Call triton kernel
    grid = lambda meta: (triton.cdiv(number_of_elements, meta['BLOCK_SIZE']),)
    _smelu_kernel_forward[grid](input, beta, output, number_of_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def _smelu_kernel_backward(
        input_pointer,
        beta: float,
        output_pointer,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
):
    """ Triton kernel SmeLU backward """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(input_pointer + offsets, mask=mask)
    gradient = tl.where(x >= beta, 1., 0.)
    gradient = tl.where(tl.abs(x) <= beta, 0.5 * (x + beta) / beta, gradient)
    # Write-back output
    tl.store(output_pointer + offsets, gradient, mask=mask)


def _smelu_triton_backward(
        input: torch.Tensor,
        beta: float = 2.
) -> torch.Tensor:
    """
    Wrapper function for SmeLU backward triton kernel
    :param input (torch.Tensor): Input tensor of any shape
    :param beta (float): Beta value of SmeLU
    :return (torch.Tensor): Gradient of SmeLU
    """
    # Init output tensor
    output: torch.Tensor = torch.empty_like(input)
    # Make input contiguous if needed
    if not input.is_contiguous():
        input = input.contiguous()
    # Get number of elements in input
    number_of_elements: int = input.numel()
    # Call triton kernel
    grid = lambda meta: (triton.cdiv(number_of_elements, meta['BLOCK_SIZE']),)
    _smelu_kernel_backward[grid](input, beta, output, number_of_elements, BLOCK_SIZE=1024)
    return output


class _SmeLU(autograd.Function):
    """
    Autograd wrapper for triton implementation.
    """

    @staticmethod
    def forward(
            ctx,
            input: torch.Tensor,
            beta: float
    ) -> torch.Tensor:
        """
        Forward pass.
        :param ctx: Context variable
        :param input (torch.Tensor): Input tensor of any shape
        :param beta (torch.Tensor): Beta value as a scalar tensor
        :return (torch.Tensor): Activation tensor of the same shape as the input
        """
        # Save input tensor and beta value for backward pass
        ctx.save_for_backward(input)
        ctx.beta = beta
        # Compute output activation
        output: torch.Tensor = _smelu_triton_forward(input=input, beta=beta)
        return output

    @staticmethod
    def backward(
            ctx,
            grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
        Backward pass.
        :param ctx: Context variable
        :param grad_output (torch.Tensor): Previous gradient
        :return (Tuple[torch.Tensor, None]): Gradient of input
        """
        # Get saved variables
        input, = ctx.saved_tensors
        beta: float = ctx.beta
        # Compute gradient
        gradient = _smelu_triton_backward(input=input, beta=beta)
        return gradient * grad_output, None


# Make autograd function
smelu_function = _SmeLU.apply


class SmeLU(nn.Module):
    """
    This class implements the Smooth ReLU (SmeLU) activation function proposed in:
    https://arxiv.org/pdf/2202.06499.pdf.
    """

    def __init__(
            self,
            beta: float = 2.
    ) -> None:
        """
        Constructor method.
        :param beta (float): Beta value if the SmeLU activation function. Default 2.
        """
        # Call super constructor
        super(SmeLU, self).__init__()
        # Check beta
        assert beta > 1e-05, f"Beta must be equal or larger than zero (> 1e-05). beta={beta} given."
        # Save parameter
        self.beta: float = beta

    def __repr__(self) -> str:
        """
        Returns a string representation.
        :return (str): String representation
        """
        return f"SmeLU(beta={self.beta})"

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        :param input (torch.Tensor): Tensor of any shape
        :return (torch.Tensor): Output activation tensor of the same shape as the input tensor
        """
        return smelu_function(input, self.beta)
