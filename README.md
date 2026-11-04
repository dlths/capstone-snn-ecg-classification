# SNN for ECG Arrhythmia Classification on a Neuromorphic Chip

**English** | [한국어](README.ko.md)

> Classifying normal vs. arrhythmic heartbeats with a Spiking Neural Network (SNN), and running inference on **Neu+**, a neuromorphic chip developed by KIST.

**Capstone Design Project, 2023** · Dept. of Electronic and IT Media Engineering, Seoul National University of Science and Technology
Team: Soeun Lee, Yunseon Choi · Advisor: Prof. Seungho Choi
📄 Paper accepted at the IEIE Fall Conference 2023 — *Recent Trends in Spike Encoding for Time-Series Data* (in Korean)

![overview](assets/overview.png)

## Highlights

- **Implemented four spike encoding methods (HSA, BSA, TTFS, Burst)** for time-series signals and compared them on the same SNN in terms of accuracy and spike efficiency.
- **Burst coding achieved the best accuracy (85.28%)** with a spike density of only 1.13%, about 1/30 of HSA.
- Since the neuromorphic chip does not support bias terms, we **retrained the model without biases** and verified inference on the actual Neu+ chip at KIST.

## Motivation

GPU-based neural networks suffer from memory bottlenecks and high power consumption as models grow. Neuromorphic chips mimic biological neurons and synapses to compute efficiently, and SNNs, which transmit information as discrete spikes, are the natural fit for them. With real-time arrhythmia detection on wearable devices in mind, our goal was to find the **spike encoding best suited to 1-D time-series signals such as ECG**.

## Method

**Data** — MIT-BIH Arrhythmia Database
- Re-detected each R-peak and cropped 50 steps on each side (100 steps), then concatenated samples to form a 300-step input.
- Merged the 5 classes (N, S, V, F, Q) into 2: normal (N) vs. arrhythmia (others).
- Balanced ~70k normal / ~15k arrhythmia samples by undersampling the normal class (30k total).

**Model** — fully connected SNN: `input 300 → LIF 50 → LIF 2`
- Leaky Integrate-and-Fire (LIF) neurons, 100 simulation time steps
- Trained with surrogate-gradient backpropagation (atan) and cross-entropy loss
- All layers use `bias=False` so that the weights can be deployed on the chip

**Spike encodings**

| Method | Type | Idea |
|---|---|---|
| HSA (Hough Spiker) | Deconvolution | Fires when the whole FIR filter window fits under the signal, then subtracts the filter |
| BSA (Ben's Spiker) | Deconvolution | Fires by comparing two accumulated error terms |
| TTFS (Time-to-First-Spike) | Temporal | Exponentially decaying threshold; larger values fire earlier, only once |
| Burst | Temporal | Threshold grows by β after each spike, producing short-interval spike bursts |

| HSA / BSA | TTFS / Burst |
|---|---|
| ![](assets/encoding_hsa_bsa.png) | ![](assets/encoding_ttfs_burst.png) |

## Results

| Encoding | Accuracy (%) | # Spikes | Spike density (%) |
|---|---:|---:|---:|
| HSA | 84.15 | 10,864 | 36.21 |
| BSA | 81.85 | 8,765.8 | 29.22 |
| TTFS | 50.00 | 202.21 | 0.674 |
| **Burst** | **85.28** | 337.93 | 1.126 |

<sub>Density = 100 × (# spikes) / (data length × time steps)</sub>

- **Burst** captures signal features with very few spikes, achieving both accuracy and efficiency.
- **TTFS** fails to classify: each input fires only once, so information fades as it propagates through layers. Too low a spike density saves energy but loses information.

**On-chip inference** — We deployed the trained weights on the Neu+ chip at the KIST Neuromorphic Computing Lab and confirmed neuron firing during inference.

![Neu+ spikes](assets/neuplus_spikes.png)

## Code

```
src/
├── encoders.py   # HSA, BSA, TTFS and Burst spike encoders
└── model.py      # SNN classifier with the Burst encoder (other models differ only in the encoder)
```

The original repository was not preserved, so the **core code (encoders and model) was restored from the final project report**. Data preprocessing and training scripts are not included.

```bash
pip install torch snntorch scipy
```

```python
from types import SimpleNamespace
import torch
from model import BurstSnn

args = SimpleNamespace(num_steps=100, data_num_steps=300, threshold=0.125,
                       beta=0.5, num_classes=2, device="cpu")
model = BurstSnn(args)
spk_out, n_input_spikes = model(torch.rand(8, 300))  # inputs normalized to [0, 1]
pred = spk_out.sum(0).argmax(1)                      # classify by output spike count
```

## Tech Stack

Python · PyTorch · snnTorch · NumPy · SciPy · NVIDIA RTX A5000

## Lessons Learned

- Biases improved accuracy in simulation, but the chip's constraints forced a bias-free redesign. This taught us to account for hardware constraints from the start.
- As future work, we proposed running larger models (Transformer, AutoEncoder) on the chip via ANN-to-SNN conversion.

## References

- Schrauwen & Van Campenhout, *BSA, a fast and accurate spike train encoding scheme*, IJCNN 2003
- Park et al., *T2FSNN: Deep spiking neural networks with time-to-first-spike coding*, DAC 2020
- Park et al., *Fast and efficient information transmission with burst spikes in deep spiking neural networks*, DAC 2019
- Kachuee et al., *ECG heartbeat classification: A deep transferable representation*, ICHI 2018
- [snnTorch documentation](https://snntorch.readthedocs.io/)
