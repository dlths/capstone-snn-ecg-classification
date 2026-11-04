"""SNN classifier for ECG (normal vs. arrhythmia).

Restored from the 2023 capstone final report. Architecture: spike encoder ->
FC(300 -> 50) -> LIF -> FC(50 -> 2) -> LIF, trained with surrogate gradients
(atan). Layers have no bias so the trained weights can be deployed on the
Neu+ neuromorphic chip, which does not support bias terms.

All four models share this structure and differ only in `self.encoder`;
the Burst version is shown here, as in the report.
"""
import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from encoders import BURST


class BurstSnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_steps = args.num_steps  # simulation time steps (100)
        spike_grad = surrogate.atan()

        self.encoder = BURST(data_num_steps=args.data_num_steps,
                             init_th=args.threshold, device=args.device)
        self.fc1 = nn.Linear(args.data_num_steps, 50, bias=False)
        self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(50, args.num_classes, bias=False)
        self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []      # output spikes per time step
        spike_count = []  # number of input spikes per time step

        for t in range(self.num_steps):
            input_spike = self.encoder(x, t)
            cur1 = self.fc1(input_spike)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
            spike_count.append(input_spike.sum(1))

        return torch.stack(spk_rec, dim=0), torch.stack(spike_count, dim=0).sum(0)
