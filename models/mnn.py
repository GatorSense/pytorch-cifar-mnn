#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt

def hitmiss(x, K_hit, K_miss, padding=0, stride=1):
    B, C_in, H, W = x.shape
    C_out, C_in_k, k, _ = K_hit.shape

    # print(f'Shape of input: {x.shape}')

    assert C_in == C_in_k, f"K_hit input channels ({C_in_k}) must match x ({C_in})"

    # Pad input
    if padding:
        x = F.pad(x, (padding,) * 4)

    # Extract patches: [B, C_in * k², N] where N = number of sliding positions
    patches = F.unfold(x, kernel_size=k, stride=stride)
    N = patches.shape[-1]  # Number of sliding positions
    patches = patches.view(B, C_in, k, k, N)  # [B, C_in, k, k, N]
    # for i in range(N):
    #     patch = patches[0, 0, :, :, i].cpu()
    #     plt.imshow(patch, cmap='gray')
    #     plt.title(f"Patch {i}")
    #     plt.show()

    # Flatten filters and broadcast
    K_hit = K_hit.view(1, C_out, C_in, k, k, 1)
    K_miss = K_miss.view(1, C_out, C_in, k, k, 1)
    p = patches.unsqueeze(1)  # [B, 1, C_in, k, k, N]

    # print(f'Shape of patches {patches.shape}')

    # Hit-or-miss morphological operation
    hit = (p - K_hit).amin(dim=(2, 3, 4))         # [B, C_out, C_in, N]
    miss = (p - K_miss).amax(dim=(2, 3, 4))       # [B, C_out, C_in, N]
    out = hit - miss     # [B, C_out, N]

    # print(f'Shape of feature maps {out.shape}')

    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1
    assert H_out * W_out == N, f"Expected {H_out}x{W_out} = {H_out * W_out}, got N={N}"

    # print(f'Inferred output shape {H_out}x{W_out}')

    # fms = out.view(B, C_out, H_out, W_out).contiguous()
    # for i in range(B):
    #     plt.imshow(fms[i][0].detach().cpu(), cmap='gray')
    #     plt.show()

    return out.view(B, C_out, H_out, W_out).contiguous()


class MNN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding=0, stride=1):
        super(MNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.K_hit = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.K_miss = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * math.pow(self.kernel_size, 2)
        stdv = 1. / math.sqrt(n)
        self.K_hit.data.uniform_(-stdv, stdv)
        self.K_miss.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = hitmiss(x, self.K_hit, self.K_miss, self.padding, self.stride)
        if torch.isnan(out).any():
            print("NaN detected in MNN output")
            print("K_hit stats:", self.K_hit.mean().item(), self.K_hit.std().item())
            print("K_miss stats:", self.K_miss.mean().item(), self.K_miss.std().item())
            raise ValueError("NaN encountered in MNN layer")
        return out
    