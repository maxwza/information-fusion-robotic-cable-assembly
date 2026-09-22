# Chapter 2 — Process-Aware Feature Modulation for Fine-Grained Connector Detection

This directory contains the implementation and supplementary materials for Chapter 2 of the dissertation.

The proposed method is an FCOS-based connector detection framework that incorporates process-related information into visual feature representation for fine-grained connector detection in robotic cable assembly.

## Method

The proposed detector consists of:

* ConvNeXt V2 backbone
* EMA attention module
* SimAM-enhanced FPN
* Process-Aware Feature Modulation Network (PFNM)
* FCOS detection head

The overall framework is:

```text
Input Image
    ↓
ConvNeXt V2
    ↓
EMA
    ↓
SimAM-FPN
    ↓
Process-Aware Feature Modulation
    ↓
FCOS Head
    ↓
Connector Detection
```

## Dataset

The dataset contains **8,000 images** from **12 connector categories**.

| Split      | Images |
| ---------- | -----: |
| Training   |  5,600 |
| Validation |  1,200 |
| Test       |  1,200 |

The complete dataset and process-related metadata are not included in this repository because of research and confidentiality restrictions.

Dataset information is provided in:

```text
datasets/README.md
```

## Code Structure

```text
chapter2_detection/
├── fcos_improved/
├── configs/
├── datasets/
├── README.md
└── requirements.txt
```

The main implementation of the proposed method is provided in `fcos_improved/`.

## Environment

The experiments were conducted using:

```text
Python 3.10
PyTorch 2.5.1
CUDA 12.1
NVIDIA RTX 4070
64 GB RAM
```

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Results

The proposed method achieved the following results on the connector detection dataset:

| Metric     | Result |
| ---------- | -----: |
| mAP        |  84.7% |
| Precision  |  92.9% |
| Recall     |  90.8% |
| F1-score   |  91.8% |
| FPS        |   17.6 |
| Parameters | 45.8 M |

These results correspond to the experimental configuration reported in Chapter 2 of the dissertation.

## Relation to the Following Chapters

The connector detection results generated in this chapter provide the visual perception input for the 6D pose estimation method presented in Chapter 3.

```text
Chapter 2
Connector Detection
      ↓
Chapter 3
6D Pose Estimation
      ↓
Chapter 4
Path Planning
      ↓
Chapter 5
Cable Assembly Execution
```


