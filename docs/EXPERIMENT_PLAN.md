# DP-Shampoo: MICCAI 2026 Experiment Plan

## Paper Framing

**Title (working):** *Shampoo Preconditioning for Differentially Private Medical Image Segmentation via Public and Synthetic Data*

**Core thesis:** Second-order gradient statistics estimated from public or synthetic data can precondition DP-SGD at zero privacy cost. We show that Shampoo — a gradient-only Kronecker preconditioner — is uniquely suited to this setting because:

1. **Robust to distribution mismatch.** Unlike K-FAC (which needs activation and backpropagation covariances that are highly data-dependent), Shampoo uses gradient outer products that primarily reflect network architecture. When the preconditioning data differs from private data, Shampoo degrades gracefully.

2. **Full layer coverage.** K-FAC requires per-layer hooks and only supports Conv2d + Linear. Shampoo covers Conv2d, ConvTranspose2d, Linear, and falls back to diagonal for GroupNorm/bias — important for U-Net decoders where ConvTranspose2d holds ~9% of parameters.

3. **No hooks required.** Shampoo works directly from `param.grad`, making it simpler to integrate with Opacus and less error-prone.

4. **Novel for DP-SGD.** To our knowledge, Shampoo has not been applied to differentially private training. AdaDPS (diagonal) exists; K-FAC for DP exists in limited settings (MNIST). Shampoo is the sweet spot between the two.

**Key narrative arc:**
- DP-SGD adds isotropic noise → geometry-aware preconditioning helps
- Diagonal (AdaDPS) captures per-parameter scale but misses cross-parameter correlations
- K-FAC captures correlations but is fragile under distribution shift
- Shampoo captures correlations through gradient outer products, which are architecture-driven and thus robust to distribution mismatch
- Synthetic data preconditioning achieves comparable gains to public data → **zero privacy cost**

**Positioning vs. prior work:**
| Method | Curvature | Data needed | Layer coverage | Distribution robustness |
|--------|-----------|-------------|----------------|------------------------|
| AdaDPS (De+2022) | Diagonal E[g²] | Public/synthetic | All | High (per-param scale) |
| K-FAC (ours, ablation) | Kronecker Fisher | Public/synthetic | Conv2d+Linear only | Low (activation-dependent) |
| **Shampoo (ours)** | **Kronecker gradient** | **Public/synthetic** | **All** | **High (gradient-only)** |

---

## Experimental Design

### Datasets

| Dataset | Task | Domain | N_train | N_val | Image size | Metric |
|---------|------|--------|---------|-------|------------|--------|
| **Kvasir-SEG** | Polyp segmentation | GI endoscopy | 800 | 200 | 256×256 | Dice |
| **Oxford Pet** | Binary segmentation | Natural images | ~2,944 | ~736 | 128×128 | Dice |
| **PathMNIST** | 9-class classification | Pathology | 89,996 | 10,004 | 28×28 | Accuracy |

**Rationale:**
- Kvasir = primary medical dataset (polyp segmentation is clinically relevant, small N makes DP challenging)
- Pet = secondary segmentation benchmark (larger N, tests generalization beyond medical domain)
- PathMNIST = classification task (shows method generalizes beyond segmentation)

### Methods

| ID | Method | Preconditioner | Estimation source | Privacy cost |
|----|--------|---------------|-------------------|--------------|
| `baseline` | SGD (no DP) | None | — | ε = ∞ |
| `dp` | DP-SGD | None | — | ε |
| `dp_adadps` | DP-SGD + AdaDPS | Diagonal | Public (oracle) | ε |
| `dp_adadps_synth` | DP-SGD + AdaDPS | Diagonal | Synthetic | ε |
| `dp_shampoo` | DP-SGD + Shampoo | Kronecker gradient | Public (oracle) | ε |
| `dp_shampoo_synth` | DP-SGD + Shampoo | Kronecker gradient | Synthetic | ε |

**Dropped:** K-FAC (underperforms AdaDPS on all datasets; mentioned briefly in paper as motivation for Shampoo).

### Model & Training

- **Architecture:** U-Net with GroupNorm, features=[32, 64, 128, 256]
- **Optimizer:** SGD with momentum (0.9), weight decay 1e-4
- **Scheduler:** Cosine annealing
- **DP parameters:** max_grad_norm=1.0, delta=1e-5 (Kvasir/Pet), delta=1/(10*N) (PathMNIST)
- **Shampoo damping:** 1e-4
- **AdaDPS damping:** 0.1
- **Estimation budget:** 1000 synthetic samples / min(100, len(loader)) steps from public data

---

## Experiments

### Experiment 1: Main Results Table (Table 1 in paper)

**Goal:** Show Shampoo outperforms AdaDPS and vanilla DP-SGD across datasets.

| | | |
|---|---|---|
| **Datasets** | Kvasir, Pet, PathMNIST | 3 |
| **Methods** | baseline, dp, dp_adadps, dp_adadps_synth, dp_shampoo, dp_shampoo_synth | 6 |
| **Epsilon** | 8.0 (fixed) | 1 |
| **Seeds** | 42, 123, 456 | 3 |
| **Epochs** | 30 (Kvasir/Pet), 30 (PathMNIST) | — |
| **Total runs** | 3 × 6 × 3 = **54** | |

**Report:** Mean ± std per (dataset, method). Bold best DP method. Show gap to baseline.

**Expected table format:**
```
Dataset    | Baseline    | DP-SGD      | AdaDPS(pub) | AdaDPS(syn) | Shampoo(pub) | Shampoo(syn)
-----------+-------------+-------------+-------------+-------------+--------------+-------------
Kvasir     | 0.60 ± 0.01| 0.48 ± 0.01 | 0.52 ± 0.01| 0.50 ± 0.01| **0.52 ± 0.01** | 0.50 ± 0.01
Pet        |             |             |             |             |              |
PathMNIST  |             |             |             |             |              |
```

---

### Experiment 2: Privacy Budget Sweep (Figure 1 or Table 2)

**Goal:** Show Shampoo's advantage grows at tighter privacy (lower ε).

| | | |
|---|---|---|
| **Datasets** | Kvasir (primary), Pet (secondary) | 2 |
| **Methods** | dp, dp_adadps, dp_shampoo, dp_shampoo_synth | 4 |
| **Epsilons** | 1.0, 2.0, 4.0, 8.0 | 4 |
| **Seeds** | 42, 123, 456 | 3 |
| **Total runs** | 2 × 4 × 4 × 3 = **96** | |

**Report:** Line plot (x=epsilon, y=Dice) with error bands. One subplot per dataset. Baseline as horizontal dashed line.

**Hypothesis:** At ε=1, the noise dominates and preconditioning matters most. Shampoo should show larger relative gain vs AdaDPS at low ε.

---

### Experiment 3: Synthetic Data Ablation (Table 3)

**Goal:** Understand what synthetic data properties matter for Shampoo estimation.

#### 3a: Noise type (rows) × Preconditioner (columns)

| | | |
|---|---|---|
| **Dataset** | Kvasir | 1 |
| **Methods** | dp_adadps_synth, dp_shampoo_synth | 2 |
| **Noise types** | white, pink, brown, perlin | 4 |
| **Epsilon** | 8.0 | 1 |
| **Seeds** | 42, 123, 456 | 3 |
| **Total runs** | 2 × 4 × 3 = **24** | |

**Report:** Table showing Dice per (noise_type, preconditioner). Identify best noise type for each.

#### 3b: Estimation budget (how many synthetic samples)

| | | |
|---|---|---|
| **Dataset** | Kvasir | 1 |
| **Method** | dp_shampoo_synth | 1 |
| **Num samples** | 100, 250, 500, 1000, 2000 | 5 |
| **Noise type** | pink (or best from 3a) | 1 |
| **Seeds** | 42, 123, 456 | 3 |
| **Total runs** | 5 × 3 = **15** | |

**Report:** Line plot (x=num_samples, y=Dice). Show diminishing returns and identify sweet spot.

#### 3c: Damping sensitivity

| | | |
|---|---|---|
| **Dataset** | Kvasir | 1 |
| **Method** | dp_shampoo (public) | 1 |
| **Damping values** | 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 | 5 |
| **Seeds** | 42, 123, 456 | 3 |
| **Total runs** | 5 × 3 = **15** | |

**Report:** Bar plot or table showing Dice vs damping. Identify robust range.

---

### Experiment 4: Convergence Analysis (Figure 2)

**Goal:** Show Shampoo accelerates convergence, not just final performance.

- Use training history from Experiment 1 (no extra runs needed)
- Plot val_dice vs epoch for all methods on Kvasir
- One figure with 6 curves (mean across seeds, shaded ± std)
- Show Shampoo reaches DP-SGD's final performance in fewer epochs

---

### Experiment 5: Public vs Synthetic Gap Analysis (Discussion)

**Goal:** Quantify how close synthetic preconditioning gets to the oracle (public data).

- Computed from Experiment 1 results (no extra runs)
- Report: gap = Dice(public) - Dice(synthetic) for both AdaDPS and Shampoo
- **Hypothesis:** Shampoo's public-synthetic gap is smaller than AdaDPS's because gradient outer products are more architecture-driven

---

## Run Budget Summary

| Experiment | Runs | GPU-hours (est. @ 3min/run Kvasir, 5min/run Pet, 2min/run PathMNIST) |
|------------|------|------|
| 1. Main table | 54 | ~4h |
| 2. Epsilon sweep | 96 | ~7h |
| 3a. Noise ablation | 24 | ~1.5h |
| 3b. Sample budget | 15 | ~1h |
| 3c. Damping sensitivity | 15 | ~1h |
| 4. Convergence (free) | 0 | 0 |
| 5. Gap analysis (free) | 0 | 0 |
| **LR tuning** | ~30 | ~1.5h |
| **Total** | **~234** | **~16h** |

---

## Implementation TODO

### Phase 0: Script updates (before any runs)
1. **Integrate Shampoo into training scripts** — Add `dp_shampoo` and `dp_shampoo_synth` methods to:
   - `scripts/train_kvasir.py`
   - `scripts/train_pet_segmentation.py`
   - `scripts/train_medmnist.py`
2. **Update `run_all_experiments.py`** — Replace K-FAC/momentum methods with Shampoo variants. Update method lists, LR dicts, and sweep methods.
3. **Update `analyze_results.py`** — Add Shampoo to result collection and table generation. Add convergence curve plotting. Add LaTeX table for ablations.
4. **Add `dp_adadps_synth` method** to training scripts (currently only dp_synthetic exists which uses AdaDPS but the naming is inconsistent — standardize to `dp_adadps_synth`).

### Phase 1: LR tuning (~30 runs)
5. Tune LR for new methods (dp_shampoo, dp_shampoo_synth, dp_adadps_synth) on each dataset.

### Phase 2: Main experiments (Exp 1 + 2 = 150 runs)
6. Run main table (54 runs).
7. Run epsilon sweep (96 runs).

### Phase 3: Ablations (Exp 3 = 54 runs)
8. Run noise type ablation (24 runs).
9. Run sample budget ablation (15 runs).
10. Run damping ablation (15 runs).

### Phase 4: Analysis & figures
11. Collect all results with `analyze_results.py`.
12. Generate main table (LaTeX).
13. Generate epsilon sweep figure.
14. Generate convergence curves.
15. Generate ablation tables.

---

## Paper Outline (6 pages + 2 refs, MICCAI format)

### 1. Introduction (0.75 pages)
- DP-SGD for medical imaging: necessary but hurts utility
- Preconditioning can help but requires curvature info
- Key insight: estimate curvature from public/synthetic data at zero privacy cost
- Contribution: Shampoo preconditioner for DP-SGD, robust to distribution mismatch

### 2. Related Work (0.5 pages)
- DP-SGD and variants (Abadi+2016, Opacus)
- Preconditioning for DP-SGD (De+2022 AdaDPS, diagonal)
- K-FAC and second-order methods (Martens+2015)
- Shampoo optimizer (Gupta+2018, Anil+2020)

### 3. Method (1.5 pages)
- 3.1 Background: DP-SGD, per-sample clipping, noise addition
- 3.2 Preconditioning before clipping (framework)
- 3.3 Shampoo preconditioner: L=E[GG^T], R=E[G^TG], apply L^{-1/4} G R^{-1/4}
- 3.4 Estimation from public/synthetic data
- 3.5 Why Shampoo > K-FAC for this setting (gradient-only = distribution-robust)
- Algorithm box: DP-SGD with Shampoo preconditioning

### 4. Experiments (2.5 pages)
- 4.1 Setup (datasets, model, hyperparameters)
- 4.2 Main results (Table 1)
- 4.3 Privacy budget analysis (Figure 1)
- 4.4 Ablations: noise type, estimation budget, damping (Table 2)
- 4.5 Convergence analysis (Figure 2)
- 4.6 Public vs synthetic gap

### 5. Discussion & Conclusion (0.75 pages)
- Shampoo recovers X% of the DP gap
- Synthetic preconditioning achieves Y% of public (oracle) performance
- Limitations: eigendecomposition cost, large layers
- Future: online Shampoo updates, 3D medical imaging

---

## Success Criteria (From-Scratch Experiments)

For a strong MICCAI submission, we need:

1. **Shampoo(public) > AdaDPS(public)** on at least 2/3 datasets (already shown on Kvasir: 0.5227 vs 0.5190)
2. **Shampoo(synthetic) ≈ Shampoo(public)** — gap < 1.5pp on average
3. **Shampoo(synthetic) > DP-SGD** — statistically significant (p < 0.05 via paired t-test across seeds)
4. **Advantage grows at low ε** — Shampoo's gain at ε=1 > gain at ε=8
5. **Robust to noise type** — reasonable performance across all synthetic noise types
6. **Reasonable estimation budget** — 500-1000 samples sufficient (no need for large public datasets)

---

## EXTENSION: DP-LoRA Finetuning with Shampoo (Experiment 6)

### Motivation

Our from-scratch U-Net results show that Shampoo preconditioning helps DP-SGD (Dice: 0.45→0.50 on Kvasir, 0.70→0.72 on Pet). However, modern medical imaging increasingly uses **pretrained foundation models with parameter-efficient finetuning (PEFT)**. We extend our approach to show Shampoo preconditioning helps across paradigms.

**Key insight:** LoRA reduces the trainable parameter space from millions to thousands, which means DP noise is injected into a much lower-dimensional space. Shampoo's role becomes even more critical: with fewer parameters, each gradient dimension carries more information, and proper preconditioning extracts maximum signal from noisy gradients.

**Novelty:** No prior work combines Shampoo + LoRA + DP-SGD. This three-way combination is novel.

### Literature Context

| Paper | Venue | Key Insight |
|-------|-------|-------------|
| LoRA Done RITE | ICLR 2025 | Matrix preconditioner for LoRA achieves transformation invariance (+7pp on GSM8K) |
| Riemannian Preconditioned LoRA | ICML 2024 | r×r preconditioner stabilizes LoRA training, robust to LR choice |
| LoRA Imitates DP-SGD | ICML 2025 | LoRA is theoretically equivalent to noisy SGD; lower rank = more implicit privacy |
| FFA-LoRA | ICLR 2024 | Freeze A, train only B → eliminates quadratic DP noise amplification in B@A |
| DP-LoRA (Diffusion) | ICCV 2025 | DP-SGD on LoRA only; outperforms full-model DP finetuning by >20% FID |
| DP in Medical DL | npj Digital Med 2025 | "Segmentation surprisingly robust to DP" — supports our paper thesis |

### Architecture

**Source:** Adapted from `/home/mmolinav/Projects/MedRadSeg/`

```
Frozen RADIO ViT-B (105M params, zero privacy cost)
  └─ LoRA on QKV in last 4 attention blocks
       lora_A: (r, 768) per layer — Kaiming init
       lora_B: (768, r) per layer — zero init
       Forward: y = Wx + scaling * (x @ A^T @ B^T)

Trainable Segmentation Head (with GroupNorm, NOT BatchNorm)
  └─ Conv2d(768, 256, 3) + GN(256) + ReLU
  └─ Upsample 4x
  └─ Conv2d(256, 64, 3) + GN(64) + ReLU
  └─ Upsample 4x
  └─ Conv2d(64, 1, 1)
```

**Parameter counts (rank=4, 4 layers):**
- LoRA: 4 × (4×768 + 768×4) = 24,576 params
- Seg_head: ~200K params
- **Total trainable: ~225K** (vs 105M frozen backbone)
- DP noise dimension reduced **~467x** vs full model finetuning

### Methods

| ID | Method | LoRA | DP | Preconditioner | What's trainable |
|----|--------|------|----|----------------|------------------|
| `lora_baseline` | Non-DP finetuning | A+B | No | None | LoRA A,B + seg_head |
| `lora_dp` | DP-SGD finetuning | A+B | Yes | None | LoRA A,B + seg_head |
| `lora_dp_shampoo` | DP-SGD + Shampoo | A+B | Yes | Shampoo (public) | LoRA A,B + seg_head |
| `lora_dp_shampoo_synth` | DP-SGD + Shampoo | A+B | Yes | Shampoo (synthetic) | LoRA A,B + seg_head |
| `lora_dp_adadps` | DP-SGD + AdaDPS | A+B | Yes | Diagonal (synthetic) | LoRA A,B + seg_head |
| `lora_dp_ffa` | FFA-LoRA + DP-SGD | B only | Yes | None | LoRA B + seg_head |
| `lora_dp_ffa_shampoo` | FFA-LoRA + Shampoo | B only | Yes | Shampoo (public) | LoRA B + seg_head |

### Datasets

- **Primary:** Kvasir-SEG polyp segmentation (same as from-scratch experiments for direct comparison)
  - Source: MedRadSeg `fedrad/data/polyp.py`, client 0
  - Image size: 352×352 (RADIO preferred resolution)
  - Non-DP baseline from MedRadSeg: ~0.90 Dice
- **Secondary:** CVC-ClinicDB (client 3 from MedRadSeg polyp benchmark)
  - Shows generalization to another polyp dataset

### Experiment Matrix

| | | |
|---|---|---|
| **Datasets** | Kvasir-SEG, CVC-ClinicDB | 2 |
| **Methods** | 7 (see above) | 7 |
| **Epsilons** | 1.0, 2.0, 4.0, 8.0 (DP methods only) | 4 |
| **Seeds** | 42, 123, 456 | 3 |
| **LoRA ranks** | 4 (primary), 8 (ablation) | 1-2 |
| **Total runs (main)** | 2×(1×3 + 6×4×3) = **150** | |
| **Total runs (rank ablation)** | 1×7×1×3 = **21** | |
| **Est. time per run** | ~3 min (forward pass is fast, few trainable params) | |
| **Total GPU-hours** | ~9h | |

### Implementation Plan

#### Step 1: Copy and adapt LoRA code
- Copy `radio_lora.py` from MedRadSeg → `src/models/radio_lora.py`
- Copy `seg_head.py` → `src/models/seg_head.py`
- **Replace BatchNorm2d → GroupNorm** in seg_head (Opacus requirement)
- Restructure LoRALinear for Opacus compatibility:
  - Replace `x @ lora_A.t() @ lora_B.t()` with `nn.Linear` layers
  - `lora_down = nn.Linear(d_in, r, bias=False)` (equivalent to lora_A)
  - `lora_up = nn.Linear(r, d_out, bias=False)` (equivalent to lora_B)
  - This ensures Opacus can compute per-sample gradients via its `nn.Linear` hook

#### Step 2: Copy dataset code
- Copy `polyp.py` from MedRadSeg → `src/data/polyp.py`
- Adapt to use albumentations transforms compatible with RADIO input ([0,1] normalization)
- Verify Kvasir-SEG data path matches existing setup

#### Step 3: Create DP-LoRA training script
- `scripts/train_lora_dp.py` — main training script
- Key design:
  1. Load RADIO backbone (frozen, torch.hub)
  2. Inject LoRA into QKV of last 4 blocks
  3. Create seg_head with GroupNorm
  4. Collect trainable params: LoRA A,B + seg_head
  5. Wrap trainable modules with `GradSampleModule`
  6. Create `PrivacyEngine` attached to optimizer
  7. Train with per-sample clipping + Gaussian noise
- Shampoo integration:
  1. Before DP training, estimate Shampoo preconditioner from public/synthetic data
  2. For LoRA A (r×d_in) and B (d_out×r): compute L, R Kronecker factors
  3. For seg_head convolutions: same Shampoo as existing U-Net code
  4. Apply preconditioning BEFORE clipping (same framework as from-scratch)

#### Step 4: Integrate Shampoo for LoRA matrices
- Extend `ShampooPreconditioner` to handle LoRA-specific layer shapes
- LoRA A matrix: (4, 768) → L=(4,4), R=(768,768) — feasible
- LoRA B matrix: (768, 4) → L=(768,768), R=(4,4) — feasible
- Seg_head conv layers: same as existing U-Net conv handling
- Estimation: run forward/backward on public polyp data or synthetic images

#### Step 5: Run experiments
- LR tuning (5 LRs × 7 methods × 1 dataset = 35 runs, ~2h)
- Main experiments (150 runs, ~7h)
- Rank ablation (21 runs, ~1h)

### Expected Results & Paper Story

**From-scratch vs Finetuning comparison (Kvasir, ε=8.0):**
```
Setting          | Non-DP    | DP-SGD    | DP+Shampoo | Gap closed
-----------------+-----------+-----------+------------+-----------
U-Net (scratch)  | 0.60      | 0.45      | 0.50       | 33%
RADIO+LoRA (ft)  | 0.90      | 0.80-85?  | 0.83-87?   | ??%
```

**Paper narrative extension:**
1. "Shampoo preconditioning helps DP-SGD from scratch" (existing results)
2. "The same principle applies to DP finetuning with LoRA" (new results)
3. "Foundation model + LoRA + Shampoo achieves practical DP segmentation" (high Dice under DP)
4. "FFA-LoRA (freeze A) further reduces noise dimension" (ablation)
5. "Synthetic data estimation works for both paradigms" (zero privacy cost claim generalizes)

**This transforms the paper from:**
"A trick that improves DP-SGD on small U-Nets"
→ "A general preconditioning framework for DP training, applicable to both from-scratch AND foundation model finetuning"

### Opacus Compatibility Notes

- **RADIO backbone**: Frozen, no per-sample gradients needed. Use `model.backbone.requires_grad_(False)`.
- **LoRA layers**: Must be `nn.Linear` (not custom functional ops) for Opacus hooks.
- **Seg_head**: GroupNorm required (replace BatchNorm2d). Upsample layers are fine (no params).
- **Ghost clipping**: Use `grad_sample_mode='ghost'` to reduce memory (no explicit per-sample grad storage).
- **Max grad norm**: 1.0 (same as from-scratch experiments).
- **Batch size**: 8 (RADIO needs ~2GB per image at 352×352; batch of 8 fits in ~20GB VRAM).

### FFA-LoRA Variant

Based on FFA-LoRA (ICLR 2024), we include a variant that **freezes lora_A and only trains lora_B**:
- Eliminates quadratic noise amplification from the B@A product
- Reduces trainable LoRA params by 50% (12K instead of 24K)
- lora_A initialized with Kaiming and frozen → acts as random projection
- lora_B initialized to zero and trained with DP-SGD
- Shampoo preconditioning applied only to B matrices and seg_head

### Rank Ablation

Test rank = {2, 4, 8, 16} on Kvasir with dp_shampoo:
- Lower rank = fewer trainable params = less DP noise = potentially better privacy-utility
- But lower rank = less expressivity
- Expected sweet spot: rank 4 or 8
- Total: 4 × 3 seeds = 12 additional runs

---

## Updated Run Budget Summary

| Experiment | Runs | GPU-hours |
|------------|------|-----------|
| 1. Main table (from-scratch) | 54 | ~4h |
| 2. Epsilon sweep (from-scratch) | 102 | ~7h |
| 3. Ablations (from-scratch) | 54 | ~3h |
| 4-5. Convergence + gap (free) | 0 | 0 |
| **6. DP-LoRA finetuning** | **150** | **~7h** |
| **6b. Rank ablation** | **21** | **~1h** |
| **LR tuning (LoRA)** | **35** | **~2h** |
| **Total** | **~416** | **~24h** |

---

## Updated Paper Outline (8 pages MICCAI format)

### 1. Introduction (0.75 pages)
- DP-SGD for medical imaging: necessary but hurts utility
- Two settings: from-scratch training AND foundation model finetuning
- Key insight: estimate curvature from public/synthetic data at zero privacy cost
- Contribution: Shampoo preconditioner for DP-SGD, works across paradigms

### 2. Related Work (0.5 pages)
- DP-SGD and variants (Abadi+2016, Opacus)
- Preconditioning for DP-SGD (De+2022 AdaDPS)
- Shampoo optimizer (Gupta+2018, Anil+2020)
- DP-LoRA and private finetuning (FFA-LoRA, DP-LoRA)
- Foundation models for medical imaging (RADIO)

### 3. Method (1.5 pages)
- 3.1 Background: DP-SGD, per-sample clipping, noise addition
- 3.2 Preconditioning before clipping (framework)
- 3.3 Shampoo preconditioner: L=E[GG^T], R=E[G^TG], apply L^{-1/4} G R^{-1/4}
- 3.4 Estimation from public/synthetic data (zero privacy cost)
- 3.5 Extension to DP-LoRA finetuning
- Algorithm box: DP-SGD with Shampoo preconditioning

### 4. Experiments (2.5 pages)
- 4.1 Setup (datasets, models, hyperparameters)
- 4.2 From-scratch results: Main table + epsilon sweep (Table 1, Figure 1)
- 4.3 **DP-LoRA finetuning results (Table 2)** — NEW
- 4.4 Ablations: noise type, estimation budget, LoRA rank (Table 3)
- 4.5 Convergence analysis (Figure 2)

### 5. Discussion & Conclusion (0.75 pages)
- Shampoo works across both training paradigms
- Foundation model + LoRA + Shampoo = practical DP medical segmentation
- Synthetic preconditioning enables zero-privacy-cost improvement
- Limitations and future work

---

## Success Criteria (Updated)

**From-scratch (existing):**
1. Shampoo > AdaDPS on 2/3 datasets ✓ (Pet: Shampoo dominates)
2. Synthetic ≈ public gap < 1.5pp ✓
3. Shampoo(synth) > DP-SGD significantly ✓

**Finetuning (new):**
7. **LoRA+Shampoo > LoRA+DP-SGD** by ≥1pp on at least 1 dataset
8. **LoRA+DP achieves >0.80 Dice** on Kvasir (practical utility under DP)
9. **FFA-LoRA competitive** with full LoRA under DP (noise reduction helps)
10. **Synthetic preconditioning works for LoRA** — gap < 2pp vs public
