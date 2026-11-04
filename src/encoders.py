"""Spike encoders for 1-D time-series (ECG) signals.

Restored from the 2023 capstone final report (Neuromorphic chip model,
Seoul National University of Science and Technology). The original
implementation called a plotting helper inside HSA; that call is removed here.

Encoders
- HSA   : Hough Spiker Algorithm (deconvolution-based)
- BSA   : Ben's Spiker Algorithm (deconvolution-based)
- TTFS  : Time-to-First-Spike coding (temporal)
- BURST : Burst coding (temporal)

HSA / BSA encode a whole batch of signals at once ([batch, length]).
TTFS / BURST are called once per simulation time step t.
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class HSA(nn.Module):
    def __init__(self, filter, new_amp=0.2, device="cuda"):
        super().__init__()
        self.device = device
        if not isinstance(filter, torch.Tensor):
            filter = torch.FloatTensor(filter)
        self.filter = filter.to(device) * new_amp

    def hsa_encode(self, sig: Tensor) -> Tensor:
        data = deepcopy(sig)
        size_input = data.shape[1]
        size_filter = self.filter.shape[0]
        self.output = torch.zeros(data.shape, device=self.device)
        for i in range(size_input - size_filter + 1):
            # fire where the whole filter window lies under the signal
            mask = torch.all(data[:, i:i + size_filter] >= self.filter, dim=1)
            data[mask, i:i + size_filter] -= self.filter
            self.output[mask, i] = 1
        self.num_spike = torch.sum(self.output, dim=1)
        return self.output.detach()

    def forward(self, input: Tensor) -> Tensor:
        return self.hsa_encode(input)

    def count_spike(self) -> Tensor:
        return self.num_spike.clone().detach()


class BSA(nn.Module):
    def __init__(self, filter, new_amp=1, threshold=0.9952, device="cuda"):
        super().__init__()
        self.threshold = threshold
        self.device = device
        if not isinstance(filter, torch.Tensor):
            filter = torch.tensor(filter)
        self.filter = filter.to(device) * new_amp

    def bsa_encode(self, sig: Tensor) -> Tensor:
        data = deepcopy(sig)
        size_input = data.shape[1]
        size_filter = self.filter.shape[0]
        self.output = torch.zeros(data.shape, device=self.device)
        for i in range(size_input - size_filter):
            error1 = abs((data[:, i:i + size_filter] - self.filter).sum(1))
            error2 = abs(data[:, i:i + size_filter].sum(1))
            mask = error1 <= (error2 * self.threshold)
            if torch.any(mask):
                self.output[mask, i] = 1
                data[mask, i + 1:i + size_filter + 1] -= self.filter
        self.num_spike = torch.sum(self.output, dim=1)
        return self.output.clone().detach()

    def forward(self, input: Tensor) -> Tensor:
        return self.bsa_encode(input)

    def count_spike(self) -> Tensor:
        return self.num_spike.clone().detach()


class TTFS(nn.Module):
    """Time-to-first-spike coding with an exponentially decaying threshold."""

    def __init__(self, data_num_steps, t_d=0, init_th=1, tau=5, device="cuda"):
        super().__init__()
        self.t_d = t_d
        self.init_th = init_th
        self.tau = tau
        self.th = 0
        self.device = device
        self.mem = torch.zeros(data_num_steps, device=device)
        self.fire_time = []

    def encoding_kernel(self, t):
        self.th = self.init_th * np.exp(-(t - self.t_d) / self.tau)

    def ttfs_encode(self, data: Tensor, t: int) -> Tensor:
        if t == 0:
            self.mem = data.clone().detach().to(self.device)
            self.th = 0
        self.encoding_kernel(t)
        if self.th >= 1e-5:
            fire = self.mem >= self.th
        else:
            fire = torch.zeros(self.mem.shape, device=self.device).bool()
        self.output = fire.float()
        # each input value fires only once
        self.mem = torch.where(fire, torch.zeros_like(self.mem), self.mem)
        self.fire_time.append(torch.where(self.output == 1))
        self.num_spike = torch.sum(self.output, dim=1)
        return self.output.clone().detach()

    def forward(self, input: Tensor, t: int) -> Tensor:
        return self.ttfs_encode(input, t)

    def count_spike(self) -> Tensor:
        return self.num_spike.clone().detach()


class BURST(nn.Module):
    """Burst coding: the threshold is multiplied by `beta` after every spike
    and reset to `init_th` when no spike occurs. Input must be in [0, 1]."""

    def __init__(self, data_num_steps, beta=2.0, init_th=0.125, device="cuda"):
        super().__init__()
        self.beta = beta
        self.init_th = init_th
        self.device = device

    def burst_encode(self, data: Tensor, t: int) -> Tensor:
        if t == 0:
            self.mem = data.clone().detach().to(self.device)
            self.th = torch.ones(self.mem.shape, device=self.device) * self.init_th
        fire = self.mem >= self.th
        self.output = fire.float()
        self.mem -= torch.where(fire, self.th, torch.zeros_like(self.mem))
        self.th = torch.where(fire, self.th * self.beta,
                              torch.ones_like(self.th) * self.init_th)
        self.num_spike = torch.sum(self.output) / self.mem.shape[0]
        return self.output.clone().detach()

    def forward(self, input: Tensor, t: int) -> Tensor:
        return self.burst_encode(input, t)
