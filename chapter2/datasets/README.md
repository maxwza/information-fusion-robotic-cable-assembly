# Connector Detection Dataset

This directory provides the dataset description and data organization information used for the connector detection experiments in Chapter 2.

The dataset was collected for fine-grained connector detection in robotic cable assembly scenarios. It contains images of industrial cable connectors acquired under representative assembly environments and is used to train and evaluate the process-aware connector detection method presented in Chapter 2.

## 1. Dataset Overview

The dataset contains **8,000 images** covering **12 connector categories**:

* C1-2
* C1-4
* C1-6
* C2-2
* C2-4
* C2-6
* C3-2
* C3-4
* C3-6
* C4-2
* C4-3
* C4-6

The dataset is designed to evaluate fine-grained connector detection under visually similar connector categories and different assembly-related conditions.

## 2. Dataset Split

The dataset is divided into training, validation, and test subsets using a group-wise split:

| Subset     | Number of Images |
| ---------- | ---------------: |
| Training   |            5,600 |
| Validation |            1,200 |
| Test       |            1,200 |
| Total      |            8,000 |

The group-wise splitting strategy is used to reduce information leakage between the training, validation, and test subsets.

## 3. Annotation

Each image is associated with object detection annotations containing the connector category and corresponding bounding-box information.

The annotations are used by the FCOS-based detection framework provided in the `fcos_improved/` directory.

The annotation format follows the format expected by the dataset loader implemented in:

```text
fcos_improved/dataset.py
```

Please refer to that file for the exact annotation parsing and preprocessing procedure.

## 4. Connector Categories

The 12 connector categories are defined as follows:

| Category | Class ID |
| -------- | -------: |
| C1-2     |        0 |
| C1-4     |        1 |
| C1-6     |        2 |
| C2-2     |        3 |
| C2-4     |        4 |
| C2-6     |        5 |
| C3-2     |        6 |
| C3-4     |        7 |
| C3-6     |        8 |
| C4-2     |        9 |
| C4-3     |       10 |
| C4-6     |       11 |

The class IDs should remain consistent with the configuration and model implementation when reproducing the experiments.

## 5. Process-Aware Information

The proposed detection method incorporates process-related information through the Process Factory Information Module.

The process information associated with the dataset includes metadata related to:

* Assembly stage
* Process parameter ranges
* Contact requirements
* Environmental conditions
* Connector category and quality information

The process-related metadata is used by the process-aware feature modulation mechanism during model inference/training.

Detailed process metadata is not included in this repository when it contains confidential or application-specific information.

## 6. Directory Structure

When the dataset is available locally, the recommended directory structure is:

```text
datasets/
├── README.md
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── annotations/
    ├── train/
    ├── val/
    └── test/
```

The exact directory names can be modified according to the configuration specified in the Chapter 2 code.

## 7. Data Availability

The complete dataset is **not included in this GitHub repository** because the images and associated process information are subject to research and confidentiality restrictions.

Researchers who require access to the complete dataset should contact the authors to discuss the availability and applicable conditions for data access.

If a publicly accessible version of the dataset becomes available, the corresponding download information will be provided here.

## 8. Reproducing the Experiments

After placing the dataset in the appropriate local directory, update the dataset paths in the configuration file:

```text
configs/
```

and verify the corresponding settings in:

```text
fcos_improved/config.py
```

The main workflow is:

```text
Dataset
   ↓
Dataset Loader
   ↓
ConvNeXt V2 Backbone
   ↓
EMA
   ↓
SimAM-FPN
   ↓
Process-Aware Feature Modulation
   ↓
FCOS Detection Head
   ↓
Connector Detection
```

Training and inference scripts are provided in the `fcos_improved/` directory.



## 9. Notes

The dataset is intended for research purposes. Users should not redistribute the dataset or associated confidential process information without appropriate authorization.

For questions regarding dataset access or experimental reproduction, please refer to the contact information provided in the main repository README.
