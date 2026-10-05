"""Linear-LDR RGB conversions. Model adapters define HDR exposure and tone mapping."""
import torch
from torch import Tensor


def srgb_to_linear(value: Tensor) -> Tensor:
    return torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(value: Tensor) -> Tensor:
    value = value.clamp(0, 1)
    return torch.where(value <= 0.0031308, value * 12.92, 1.055 * value.pow(1 / 2.4) - 0.055)
