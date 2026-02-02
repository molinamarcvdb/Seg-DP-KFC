# Dataset Setup for DP-AdaDPS Experiments

This document describes the dataset configuration for our MICCAI paper on **Differentially Private Medical Image Segmentation with AdaDPS Preconditioning**.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Dataset Requirements](#dataset-requirements)
4. [Experimental Datasets](#experimental-datasets)
   - [2D Classification](#2d-classification)
   - [2D Segmentation](#2d-segmentation)
   - [3D Classification](#3d-classification)
   - [3D Segmentation](#3d-segmentation)
5. [Private-Public Pairings](#private-public-pairings)
6. [Download Instructions](#download-instructions)
7. [Preprocessing Guidelines](#preprocessing-guidelines)
8. [Experimental Matrix](#experimental-matrix)

---

## Overview

### Research Goal

Train medical image models with **Differential Privacy (DP)** guarantees while maintaining utility through **AdaDPS preconditioning**. The key insight is that public auxiliary data can be used to estimate gradient statistics (E[g²]) that improve the efficiency of gradient clipping in DP-SGD.

### Methods Compared

| Method | DP Enabled | Preconditioning | Description |
|--------|------------|-----------------|-------------|
| `baseline` | No | None | Upper bound on utility (no privacy) |
| `dp_no_precond` | Yes | None | Standard DP-SGD |
| `dp_adadps_public` | Yes | From public data | **Our proposed method** |
| `dp_adadps_oracle` | Yes | From private data | Oracle upper bound (not truly private) |

---

## Key Concepts

### Why Two Datasets?

```
┌─────────────────────────────────────────────────────────────────┐
│                        AdaDPS Framework                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PUBLIC DATA                      PRIVATE DATA                 │
│   (No privacy constraints)         (Protected by DP)            │
│                                                                 │
│   ┌─────────────────┐              ┌─────────────────┐         │
│   │  Estimate E[g²] │              │  Train model    │         │
│   │  (gradient      │              │  with DP-SGD    │         │
│   │   statistics)   │              │                 │         │
│   └────────┬────────┘              └────────┬────────┘         │
│            │                                │                   │
│            │    ┌──────────────────┐        │                   │
│            └───►│  Preconditioner  │◄───────┘                   │
│                 │  P = 1/√(E[g²])  │                            │
│                 └────────┬─────────┘                            │
│                          │                                      │
│                          ▼                                      │
│                 ┌──────────────────┐                            │
│                 │  Better gradient │                            │
│                 │  clipping        │                            │
│                 └──────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Private Data
- **What**: The sensitive medical data we want to protect
- **How used**: Training with DP-SGD (gradient clipping + noise)
- **Privacy**: Full DP guarantees (ε, δ)

### Public Data
- **What**: Publicly available medical images (no privacy constraints)
- **How used**: ONLY for estimating preconditioner E[g²] BEFORE training
- **Privacy**: Does not consume privacy budget
- **Requirements**:
  - Same or similar modality as private data
  - Labels required (to compute gradients via loss)
  - Does NOT need to be the same task

---

## Dataset Requirements

### For Differential Privacy to Work Well

DP-SGD adds noise proportional to `σ/√n` where:
- `σ` = noise multiplier (determined by ε, δ, epochs)
- `n` = number of training samples

**Rule of thumb**: Need **thousands** of samples for good convergence under DP.

| Dataset Size | DP Feasibility | Notes |
|--------------|----------------|-------|
| < 100 | Poor | Noise overwhelms signal |
| 100 - 1,000 | Marginal | High ε needed (ε > 8) |
| 1,000 - 10,000 | Good | Moderate ε feasible (ε ~ 2-4) |
| > 10,000 | Excellent | Low ε possible (ε < 1) |

### For Preconditioner Estimation

The public dataset should:
1. **Same modality**: X-ray → X-ray, MRI → MRI, CT → CT
2. **Similar image statistics**: Resolution, contrast, anatomy
3. **Have labels**: Required to compute loss and gradients
4. **Size**: ~100-1000 samples sufficient for stable E[g²] estimation

---

## Experimental Datasets

### 2D Classification

#### PathMNIST (Pathology)

| Attribute | Value |
|-----------|-------|
| **Size** | 107,180 images |
| **Classes** | 9 (colon pathology types) |
| **Resolution** | 28×28 (resized from patches) |
| **Source** | NCT-CRC-HE-100K |
| **License** | CC BY 4.0 |

```python
# Download via MedMNIST
from medmnist import PathMNIST
train_dataset = PathMNIST(split='train', download=True)
```

#### DermaMNIST (Dermatology)

| Attribute | Value |
|-----------|-------|
| **Size** | 10,015 images |
| **Classes** | 7 (skin lesion types) |
| **Resolution** | 28×28 |
| **Source** | HAM10000 |
| **License** | CC BY-NC 4.0 |

#### BloodMNIST (Hematology)

| Attribute | Value |
|-----------|-------|
| **Size** | 17,092 images |
| **Classes** | 8 (blood cell types) |
| **Resolution** | 28×28 |
| **Source** | Blood Cell Images |
| **License** | CC BY 4.0 |

#### ChestMNIST (Radiology)

| Attribute | Value |
|-----------|-------|
| **Size** | 112,120 images |
| **Classes** | 14 (multi-label) |
| **Resolution** | 28×28 |
| **Source** | NIH ChestX-ray14 |
| **License** | CC0 1.0 |

#### OrganAMNIST / OrganCMNIST / OrganSMNIST (CT)

| Attribute | Value |
|-----------|-------|
| **Size** | ~58,000 images each |
| **Classes** | 11 (abdominal organs) |
| **Resolution** | 28×28 |
| **Source** | Liver Tumor Segmentation (LiTS) |
| **License** | CC BY 4.0 |
| **Variants** | Axial (A), Coronal (C), Sagittal (S) |

#### CheXpert (Chest X-ray - Full Resolution)

| Attribute | Value |
|-----------|-------|
| **Size** | 224,316 images |
| **Classes** | 14 (multi-label) |
| **Resolution** | Variable (up to 2320×2828) |
| **Source** | Stanford Hospital |
| **License** | Stanford CheXpert License |
| **URL** | https://stanfordmlgroup.github.io/competitions/chexpert/ |

#### Diabetic Retinopathy (Fundus)

| Attribute | Value |
|-----------|-------|
| **Size** | ~35,000 images |
| **Classes** | 5 (DR severity grades) |
| **Resolution** | Variable |
| **Source** | EyePACS / Kaggle |
| **License** | Kaggle Competition |
| **URL** | https://www.kaggle.com/c/diabetic-retinopathy-detection |

---

### 2D Segmentation

#### Kvasir-SEG (Polyp Segmentation)

| Attribute | Value |
|-----------|-------|
| **Size** | 1,000 images |
| **Task** | Binary segmentation (polyp) |
| **Resolution** | Variable (332×487 to 1920×1072) |
| **Source** | Vestre Viken Health Trust |
| **License** | CC BY 4.0 |
| **URL** | https://datasets.simula.no/kvasir-seg/ |

#### CVC-ClinicDB (Polyp Segmentation)

| Attribute | Value |
|-----------|-------|
| **Size** | 612 images |
| **Task** | Binary segmentation (polyp) |
| **Resolution** | 384×288 |
| **Source** | Hospital Clinic Barcelona |
| **License** | Research use |
| **URL** | https://polyp.grand-challenge.org/CVCClinicDB/ |

#### ISIC Skin Lesion Segmentation

| Attribute | Value |
|-----------|-------|
| **Size** | ~2,500 images (varies by year) |
| **Task** | Binary segmentation (lesion) |
| **Resolution** | Variable |
| **Source** | ISIC Archive |
| **License** | CC BY-NC 4.0 |
| **URL** | https://challenge.isic-archive.com/ |

#### Retinal Vessel Segmentation (DRIVE + STARE + HRF)

| Dataset | Size | Resolution | Source |
|---------|------|------------|--------|
| DRIVE | 40 | 565×584 | Netherlands |
| STARE | 20 | 700×605 | USA |
| HRF | 45 | 3504×2336 | Germany |
| **Total** | **105** | - | - |

> **Note**: This combined dataset is too small for DP. Use only for preliminary experiments or as supplementary results.

---

### 3D Classification

#### OrganMNIST3D

| Attribute | Value |
|-----------|-------|
| **Size** | 1,743 volumes |
| **Classes** | 11 (abdominal organs) |
| **Resolution** | 28×28×28 |
| **Source** | LiTS |
| **License** | CC BY 4.0 |

#### NoduleMNIST3D

| Attribute | Value |
|-----------|-------|
| **Size** | 1,633 volumes |
| **Classes** | 2 (nodule present/absent) |
| **Resolution** | 28×28×28 |
| **Source** | LIDC-IDRI |
| **License** | CC BY 4.0 |

#### VesselMNIST3D

| Attribute | Value |
|-----------|-------|
| **Size** | 1,909 volumes |
| **Classes** | 2 (vessel/background) |
| **Resolution** | 28×28×28 |
| **Source** | IntrA |
| **License** | CC BY 4.0 |

#### ADNI (Alzheimer's Disease)

| Attribute | Value |
|-----------|-------|
| **Size** | ~2,000 subjects |
| **Classes** | 3-4 (CN, MCI, AD) |
| **Resolution** | ~256×256×170 |
| **Source** | ADNI Consortium |
| **License** | Restricted (requires application) |
| **URL** | https://adni.loni.usc.edu/ |

---

### 3D Segmentation

#### BraTS (Brain Tumor Segmentation)

| Attribute | Value |
|-----------|-------|
| **Size** | ~2,000 cases (2021) |
| **Task** | Multi-class (ET, TC, WT) |
| **Modalities** | T1, T1ce, T2, FLAIR |
| **Resolution** | 240×240×155 |
| **Source** | RSNA-ASNR-MICCAI |
| **License** | CC BY 4.0 |
| **URL** | https://www.synapse.org/#!Synapse:syn27046444 |

#### AMOS (Abdominal Multi-Organ)

| Attribute | Value |
|-----------|-------|
| **Size** | 500 CT + 100 MRI |
| **Task** | 15 organ segmentation |
| **Resolution** | Variable |
| **Source** | Multi-center |
| **License** | CC BY 4.0 |
| **URL** | https://amos22.grand-challenge.org/ |

#### ACDC (Cardiac)

| Attribute | Value |
|-----------|-------|
| **Size** | 150 patients |
| **Task** | 3 structures (LV, RV, Myo) |
| **Resolution** | ~256×256×10 |
| **Source** | University Hospital of Dijon |
| **License** | CC BY 4.0 |
| **URL** | https://www.creatis.insa-lyon.fr/Challenge/acdc/ |

---

## Private-Public Pairings

### Recommended Pairings

The following table shows recommended private-public dataset pairings for AdaDPS experiments:

#### 2D Classification

| Private Dataset | Public Dataset | Domain Match | Notes |
|-----------------|----------------|--------------|-------|
| PathMNIST (107k) | **PatchCamelyon (327k)** | Exact | Both histopathology |
| PathMNIST (107k) | TCGA patches | Related | Different cancer types |
| DermaMNIST (10k) | **Fitzpatrick17k** | Exact | Both dermoscopy |
| DermaMNIST (10k) | ISIC 2016 | Exact | Earlier challenge year |
| BloodMNIST (17k) | **BCCD (12k)** | Exact | Blood smear microscopy |
| ChestMNIST (112k) | **NIH ChestX-ray14** | Exact | Both chest X-ray |
| CheXpert (224k) | NIH ChestX-ray14 | Exact | Different hospitals |
| Diabetic Retinopathy | **Messidor-2 (1.7k)** | Exact | Fundus images |
| Diabetic Retinopathy | APTOS 2019 | Exact | Kaggle competition |

#### 2D Segmentation

| Private Dataset | Public Dataset | Domain Match | Notes |
|-----------------|----------------|--------------|-------|
| Kvasir-SEG (1k) | **CVC-ClinicDB (612)** | Exact | Both polyp segmentation |
| Kvasir-SEG (1k) | CVC-ColonDB (380) | Exact | Colonoscopy |
| ISIC Segmentation | **ISIC 2016** | Exact | Earlier year as public |
| ISIC Segmentation | HAM10000 | Exact | Base dataset |
| Lung Segmentation | **JSRT (247)** | Exact | Japanese chest X-ray |
| Retinal Vessels | CHASE_DB1 (28) | Exact | Different source |

#### 3D Classification

| Private Dataset | Public Dataset | Domain Match | Notes |
|-----------------|----------------|--------------|-------|
| OrganMNIST3D (1.7k) | **Medical Decathlon** | Related | Various CT tasks |
| NoduleMNIST3D (1.6k) | **LUNA16 (888)** | Exact | Lung nodules |
| VesselMNIST3D (1.9k) | **IXI MRA subset** | Related | Brain MRA |
| ADNI (2k) | **OASIS (2k+)** | Related | Brain MRI, healthy |

#### 3D Segmentation

| Private Dataset | Public Dataset | Domain Match | Notes |
|-----------------|----------------|--------------|-------|
| BraTS 2021 (2k) | **BraTS 2018** | Exact | Previous challenge year |
| BraTS 2021 (2k) | IXI | Related | Healthy brains (no tumor) |
| AMOS (600) | **BTCV (50)** | Related | Abdominal CT |
| AMOS (600) | Medical Decathlon | Related | Various organs |
| ACDC (150) | **M&Ms Challenge** | Exact | Cardiac MRI |

### Domain Shift Analysis

For research purposes, we study three levels of domain match:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Domain Match Spectrum                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  EXACT MATCH          RELATED DOMAIN         DISTANT DOMAIN    │
│  ────────────         ──────────────         ──────────────    │
│  Same task            Same modality          Different modality│
│  Same modality        Different task         Different task    │
│                                                                 │
│  Example:             Example:               Example:           │
│  PathMNIST → PCam     BraTS → IXI            PathMNIST → X-ray │
│  (both histopath)     (brain MRI, no tumor)  (different domain)│
│                                                                 │
│  Expected benefit:    Expected benefit:      Expected benefit:  │
│  HIGH                 MEDIUM                 LOW/NONE           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Download Instructions

### MedMNIST (All 2D/3D MNIST-style datasets)

```bash
pip install medmnist

# Python usage
import medmnist
from medmnist import INFO

# List all available datasets
print(medmnist.INFO.keys())

# Download specific dataset
from medmnist import PathMNIST, DermaMNIST, ChestMNIST
from medmnist import OrganMNIST3D, NoduleMNIST3D

# 2D datasets
train_data = PathMNIST(split='train', download=True, root='./data')
val_data = PathMNIST(split='val', download=True, root='./data')
test_data = PathMNIST(split='test', download=True, root='./data')

# 3D datasets
train_3d = OrganMNIST3D(split='train', download=True, root='./data')
```

### PatchCamelyon (PCam)

```bash
# Via TensorFlow Datasets
pip install tensorflow-datasets
```

```python
import tensorflow_datasets as tfds
dataset = tfds.load('patch_camelyon', split='train')
```

Or direct download:
```bash
# From GitHub
wget https://github.com/basveeling/pcam/raw/master/pcam_labels.csv
# Images from Google Drive (see repo for links)
```

### Kvasir-SEG

```bash
# Direct download
wget https://datasets.simula.no/downloads/kvasir-seg.zip
unzip kvasir-seg.zip -d ./data/kvasir-seg
```

### CVC-ClinicDB

```bash
# Requires registration at grand-challenge.org
# After registration:
wget <provided_link> -O cvc-clinicdb.zip
unzip cvc-clinicdb.zip -d ./data/cvc-clinicdb
```

### BraTS 2021

```bash
# Requires Synapse account
# 1. Register at https://www.synapse.org/
# 2. Join BraTS challenge
# 3. Download via synapse client

pip install synapseclient
synapse get -r syn27046444
```

### NIH ChestX-ray14

```bash
# Direct download (large: ~42GB)
wget https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

# Or use kaggle
pip install kaggle
kaggle datasets download -d nih-chest-xrays/data
```

### OASIS (Brain MRI)

```bash
# OASIS-3: Requires application
# Visit: https://www.oasis-brains.org/

# OASIS-1 (older, smaller) - direct download available
wget https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc1.tar.gz
```

### IXI Dataset

```bash
# Direct download
wget https://brain-development.org/ixi-dataset/

# Specific modalities
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
wget http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-MRA.tar
```

### Fitzpatrick17k

```bash
# From GitHub repository
git clone https://github.com/mattgroh/fitzpatrick17k
# Follow instructions in repo for image download
```

---

## Preprocessing Guidelines

### Standard Preprocessing Pipeline

```python
# Example preprocessing for 2D images
from torchvision import transforms

def get_preprocessing(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats or compute from data
            std=[0.229, 0.224, 0.225]
        )
    ])
```

### Modality-Specific Notes

| Modality | Preprocessing Notes |
|----------|---------------------|
| **Histopathology** | Stain normalization (Macenko, Vahadane) |
| **Chest X-ray** | CLAHE, lung field normalization |
| **Dermoscopy** | Hair removal, color constancy |
| **Fundus** | Green channel extraction, CLAHE |
| **CT** | Windowing (HU), resampling to isotropic |
| **MRI** | Bias field correction, skull stripping, intensity normalization |

### Preconditioner Estimation

For estimating E[g²] from public data:

```python
def estimate_preconditioner(model, public_loader, num_steps=100):
    """
    Estimate diagonal preconditioner from public data.

    Args:
        model: Neural network
        public_loader: DataLoader for public dataset
        num_steps: Number of batches to use for estimation

    Returns:
        Dictionary mapping parameter names to E[g²] estimates
    """
    grad_sq_sum = {name: torch.zeros_like(param)
                   for name, param in model.named_parameters()}
    count = 0

    model.train()
    for i, (images, labels) in enumerate(public_loader):
        if i >= num_steps:
            break

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sq_sum[name] += param.grad ** 2

        model.zero_grad()
        count += 1

    # Compute E[g²]
    E_g_sq = {name: g_sq / count for name, g_sq in grad_sq_sum.items()}

    return E_g_sq
```

---

## Experimental Matrix

### Main Experiments (For Paper)

| ID | Type | Private | Public | Size | Primary RQ |
|----|------|---------|--------|------|------------|
| E1 | 2D Clf | PathMNIST | PCam | 107k / 327k | RQ1: Does AdaDPS help? |
| E2 | 2D Seg | Kvasir-SEG | CVC-ClinicDB | 1k / 612 | RQ2: Segmentation |
| E3 | 3D Seg | BraTS 2021 | BraTS 2018 | 2k / 1.2k | RQ3: 3D + exact match |
| E4 | 3D Seg | BraTS 2021 | IXI | 2k / 500 | RQ4: Domain shift |

### Ablation Studies

| ID | Study | Datasets | Variables |
|----|-------|----------|-----------|
| A1 | Public data size | PathMNIST + PCam | 100, 500, 1k, 5k, full |
| A2 | Domain shift | BraTS + {BraTS'18, IXI, OASIS} | Match level |
| A3 | Privacy budget | PathMNIST + PCam | ε ∈ {0.5, 1, 2, 4, 8} |
| A4 | Estimation steps | PathMNIST + PCam | steps ∈ {10, 50, 100, 500} |

### Supplementary Experiments

| ID | Type | Private | Public | Notes |
|----|------|---------|--------|-------|
| S1 | 2D Clf | DermaMNIST | Fitzpatrick17k | Skin lesions |
| S2 | 2D Clf | ChestMNIST | NIH ChestX-ray14 | Chest X-ray |
| S3 | 3D Clf | OrganMNIST3D | Medical Decathlon | CT organs |
| S4 | 2D Seg | Retinal Vessels | CHASE_DB1 | Small dataset study |

---

## Directory Structure

```
data/
├── raw/                          # Original downloaded data
│   ├── medmnist/
│   ├── pcam/
│   ├── kvasir-seg/
│   ├── cvc-clinicdb/
│   ├── brats2021/
│   ├── brats2018/
│   └── ixi/
├── processed/                    # Preprocessed data
│   ├── pathmnist/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── pcam/
│   ├── kvasir/
│   ├── cvc/
│   ├── brats21/
│   ├── brats18/
│   └── ixi/
└── splits/                       # Train/val/test splits
    ├── pathmnist_splits.json
    ├── kvasir_5fold.json
    └── brats_splits.json
```

---

## References

### Datasets

1. **MedMNIST**: Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification", Scientific Data, 2023
2. **PatchCamelyon**: Veeling et al., "Rotation Equivariant CNNs for Digital Pathology", MICCAI 2018
3. **Kvasir-SEG**: Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MMM 2020
4. **BraTS**: Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark", IEEE TMI 2015
5. **CheXpert**: Irvin et al., "CheXpert: A Large Chest Radiograph Dataset", AAAI 2019

### Methods

1. **DP-SGD**: Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
2. **AdaDPS**: Amid et al., "Private Adaptive Optimization with Side Information", ICML 2022
3. **Opacus**: Yousefpour et al., "Opacus: User-Friendly Differential Privacy Library in PyTorch", NeurIPS 2021 Workshop

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-30 | 1.0 | Initial documentation |

