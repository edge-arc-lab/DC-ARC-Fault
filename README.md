# Edge AI for DC Arc Fault Detection

DC series arc faults are a critical hazard for photovoltaic installations and one of the leading causes of fires. This work addresses the problem within the timing and energy constraints of the **UL 1699B** standard (and its international counterpart **IEC 63027**) by designing, quantizing and validating on real **STM32** hardware two complementary architectures:

1. **InceptionTime:** a deep temporal convolutional network optimized for microcontrollers with a dedicated NPU (**STM32N6570-DK** with Neural-ART NPU).
2. **MS-RCFE + Ridge:** a multi-scale extractor with 9 fixed random convolutional branches and a linear Ridge classifier implemented in C, designed for general-purpose microcontrollers without accelerators (**STM32H7S78-DK** with a 600 MHz Cortex-M7).

## Key Results

- **Detection Rate (DR):** between **98.5%** and **99.7%** across all variants (FP32 and INT8).
- **False-Positive Rate (FPR):** between **0.11%** and **0.61%**.
- **Inference latency measured on hardware:**
  - **InceptionTime INT8:** **19.73 ms** on the Neural-ART NPU (**~315×** speed-up over 6218 ms on the CPU).
  - **MS-RCFE INT8:** **69.77 ms** on a bare Cortex-M7 (RAM: **≈ 141 KiB**, Flash: **30.8 KB**); **17.78 ms** for the 3-branch pruned variant (`b3`).
- **Detection delay and energy (UL 1699B):**
  - Median detection delay: **117 – 143 ms** (worst case $1.94\text{ s} < 2.5\text{ s}$).
  - Energy dissipated until extinction: median **9.3 – 13.6 J**, absolute maximum **267.7 J** ($< 750\text{ J}$).

---

# Project Objective

The objective is to develop a classification system able to identify DC arc faults in photovoltaic installations while jointly optimizing:

- model accuracy
- robustness against data leakage
- compatibility with embedded systems
- real-time inference on edge hardware

---

# Repository Structure

```text
DC-ARC-Fault/
│
├── dataset/
│   ├── dataset/                          # Compressed 10 kHz dataset (from Kaggle)
│   ├── dataset_new/                      # Final windowed dataset and train/test split
│   │   ├── arc_dataset_new.npz           # Full windowed dataset (X, y)
│   │   ├── arc_dataset_meta_new.csv      # Metadata of all time windows
│   │   ├── arc_dataset_skipped_new.csv   # Discarded files and reason for exclusion
│   │   ├── arc_dataset_train.npz         # Training set (80% of files, no leakage)
│   │   ├── arc_dataset_meta_train.csv    # Training set metadata
│   │   ├── arc_dataset_test.npz          # Test set (20% of files, no leakage)
│   │   ├── arc_dataset_meta_test.csv     # Test set metadata
│   │   └── split_report.txt              # Split traceability report
│   │
│   ├── extraction_and_subsampling/
│   │   └── mat_tools.py                  # ETL from IEEE DataPort (250 kHz) to Kaggle (10 kHz)
│   │
│   ├── build_dataset_new.py              # Sliding-window generation (100 ms) and labeling
│   ├── split_dataset.py                  # GroupShuffleSplit (80/20) by source file
│   └── dataset_leakage_check.py          # Anti-leakage validation and audit
│
├── script/
│   ├── causal_current_norm/              # Causal normalization and UL 1699B delay test
│   │   ├── causal_norm.py                # Causal moving-window I_nom estimation with gating
│   │   └── detection_delay.py            # Detection delay measurement (K confirmations)
│   │
│   ├── voltage_label_creation/           # Label robustness and voltage-based onset analysis
│   │   ├── voltage_label_robustness.py   # t1_V parameter sweep and level-only baseline
│   │   ├── voltage_label_evaluation.py   # Model evaluation by current-level bands
│   │   ├── label_sensitivity.py          # Ground-truth parameter sensitivity (95% bootstrap)
│   │   └── delay_analysis.py             # Diagnostics of edge cases with large delays
│   │
│   ├── energy_analysis/                  # Arc energy computation (Joule)
│   │   └── arc_energy.py                 # v*i integration, real model delays and relay actuation
│   │
│   ├── variable_load_analysis/           # Variable-load robustness and spectral decomposition
│   │   ├── variable_load.py              # Impact of dynamic load changes
│   │   ├── load_fusion.py                # Joint evaluation on heterogeneous load scenarios (file A + file B)
│   │   ├── level_free_detection.py       # Detection based exclusively on AC patterns (waveform/FFT)
│   │   ├── mean_spectrum.py              # Mean power spectral density by time band
│   │   ├── multichannel_analysis.py      # Multichannel framework (current, voltage, power)
│   │   └── channel_plot.py               # Waveform visualization
│   │
│   ├── test_set_analysis.py              # Global test set evaluation, metrics and bootstrap
│   ├── make_paper_figures.py             # Automatic generation of summary figures
│   │
│   ├── training/
│   │   ├── inception_time/
│   │   │   ├── results/
│   │   │   └── train_inceptiontime.py    # Training, ONNX export and calibration
│   │   │
│   │   ├── msrcfe/
│   │   │   ├── results/
│   │   │   ├── export_msrcfe/
│   │   │   ├── reduction_results/        # Reduced variants (b3, b4, full) for ST Edge AI
│   │   │   ├── train_msrcfe.py           # MS-RCFE + Ridge training
│   │   │   ├── export_msrcfe.py          # ONNX export and calibration
│   │   │   ├── msrcfe_prune.py           # Branch ablation and feature ranking
│   │   │   ├── msrcfe_combinations.py    # Analysis of the 511 branch combinations (Pareto)
│   │   │   ├── msrcfe_reductions.py      # ONNX generation of reduced b3/b4 variants
│   │   │   └── msrcfe_seed.py            # Stability test over multiple random seeds
│   │   │
│   │   └── multirockethydra/
│   │       ├── train_arc_compare.py      # Comparative Ridge/Hydra/ArcNet training
│   │       └── results/                  # Per-model results (arcnet, hydra, ridge) and comparisons
│   │
│   └── quantized_models/
│       ├── Inception/
│       │   ├── model_comparison.py       # FP32 vs INT8 benchmark on the test set
│       │   └── inceptiontime_PerChannel_quant_calibration_data_npz_1.onnx
│       │
│       └── msrcfe/
│           ├── stm32_float/              # FP32 latency and resource reports
│           ├── stm32_int/                # INT8 latency and resource reports
│           ├── model_comparison.py       # FP32 vs INT8 + Ridge benchmark on the test set
│           ├── conversion_probe.py       # QDQ node analysis and I/O cost separation
│           ├── resource_measurement.py   # Latency benchmark and Flash/RAM estimation
│           ├── msrcfe_PerChannel_quant_calibration_msrcfe_npz_1.onnx
│           └── msrcfe_unico_PerChannel_quant_calibration_msrcfe_npz_1.onnx
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/edge-arc-lab/DC-ARC-Fault.git
cd DC-ARC-Fault
```

## 2. Create a virtual environment

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

## 3. Install the dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

The project is based on the public reference library:

**Photovoltaic (PV) DC Arc-Fault Library**  
[IEEE DataPort](https://ieee-dataport.org/open-access/photovoltaic-pv-dc-arc-library)

The original raw dataset (hundreds of GB at 250 kHz with spurious auxiliary files) is reduced and cleaned with a dedicated tool:

- `dataset/extraction_and_subsampling/mat_tools.py`:
  - `clean`: discards auxiliary files and replicates the directory hierarchy keeping only the `.mat` files.
  - `extract`: extracts the physical variables of interest (**CurrentData**, **VoltageData**, **power**), downsamples from 250 kHz to 10 kHz (decimation factor 25) and saves the compressed files after verifying their integrity.

The resulting cleaned dataset is available for direct download on Kaggle:  
[Kaggle: DC Arc Fault Dataset](https://www.kaggle.com/datasets/yassirsuarez/dc-arc-fault)

Download the archive from Kaggle and extract it into `dataset/dataset/`. Expected structure:

```text
dataset/
└── dataset/
    ├── E001_0050V_00.50A_G001
    ├── E001_0050V_00.50A_G002
    └── ...
```

---

# 1. Preprocessing and Physical Split

Starting from the signals sampled at 10 kHz in `dataset/dataset/`, dataset generation and splitting are performed by two sequential scripts.

### Window generation (`build_dataset_new.py`)

A **sliding-window** strategy emulates real-time embedded inference:

- **Time window ($W$):** 100 ms (1,000 samples at 10 kHz).
- **Step ($S$):** 50 ms (500 samples, 50% overlap).
- **Arc onset detection ($t_1$):** the nominal current $I_{nom}$ is computed over the first 200 ms, and $t_1$ is the instant at which the current falls below 95% of $I_{nom}$ for at least 300 consecutive ms.
- **Supervised labeling:**
  - `label = 0` (pre-arc / normal): windows preceding $t_1$.
  - `label = 1` (arc in progress): windows following $t_1$.
- **Balancing:** a maximum number of windows per class per file (`MAX_WIN_PER_CLASS`) limits over-representation.

Outputs in `dataset/dataset_new/`:

- `arc_dataset_new.npz`: full dataset (34,931 windows in total, tensor $X$ and labels $y$).
- `arc_dataset_meta_new.csv`: per-window traceable metadata (source file, interval $t_{start}-t_{end}$, $t_1$, nominal current, label).
- `arc_dataset_skipped_new.csv`: log of discarded `.mat` files with the reason (for example, nominal current too low).

```bash
cd dataset
python build_dataset_new.py dataset/ --out dataset_new/
```

### Leakage-free physical split (`split_dataset.py`)

The 80% train / 20% test partition uses **`GroupShuffleSplit`**, grouping by source file (`filename`). All time windows originating from the same physical experiment therefore remain in the same split, eliminating the risk of temporal leakage.

Outputs in `dataset/dataset_new/`:

- `arc_dataset_train.npz` (27,770 windows from 856 files: 10,400 pre-arc, 17,370 arc).
- `arc_dataset_test.npz` (7,161 windows from 215 files: 2,641 pre-arc, 4,520 arc).
- `arc_dataset_meta_train.csv` and `arc_dataset_meta_test.csv`: metadata tables for each split.
- `split_report.txt`: full report listing all files assigned to the test set.

```bash
python split_dataset.py dataset_new/arc_dataset_new.npz dataset_new/arc_dataset_meta_new.csv --test-size 0.20 --out dataset_new/
```

---

# 2. Data Leakage Verification

The script `dataset/dataset_leakage_check.py` runs a model-independent audit suite before any training:

1. **Class distribution:** checks the balance between classes 0 and 1 in train and test.
2. **Exact duplicate check:** verifies that no test sample $X$ coincides exactly with a training sample.
3. **Minimum Euclidean distance:** quantifies the minimum separation between train and test points.
4. **Source file isolation:** formally verifies that the intersection between train and test source files is empty ($\text{Train} \cap \text{Test} = \emptyset$).
5. **Adversarial validation:** estimates the memorization risk with a linear Ridge classifier.

```bash
python dataset_leakage_check.py dataset_new/arc_dataset_train.npz dataset_new/arc_dataset_test.npz --meta-train dataset_new/arc_dataset_meta_train.csv --meta-test dataset_new/arc_dataset_meta_test.csv
```

---

# 3. Training and Feature Extraction

The folder `script/training/` contains the training pipelines for three architectures.

## Models

| Model | Literature name | Type | STM32 deployment |
|---|---|---|---|
| InceptionTime | InceptionTime | End-to-end deep learning (trained temporal CNN) | Yes (ONNX to ST Edge AI) |
| MS-RCFE + Ridge | MS-RCFE | Multi-scale fixed-kernel ROCKET with linear classifier | Yes (ONNX to ST Edge AI) |
| MultiRocket and variants | ROCKET / MultiRocket | Feature-based (random kernels with classical ML) | Experimental only |

---

## InceptionTime

`train_inceptiontime.py` provides, in a single script:

- training with tsai and PyTorch on GPU
- UL 1699B metrics (detection rate, false positive rate)
- multi-threshold analysis
- export of `inceptiontime.onnx` (dynamic shape)
- generation of the calibration dataset for INT8 quantization

**Output files:**

```text
inception_results/
├── inceptiontime.onnx            # for ST Edge AI
├── inceptiontime_training.png    # loss/accuracy curves
├── results_inceptiontime.png     # confusion matrix, ROC, PR, score distribution
├── inceptiontime_report.txt      # full metrics
├── calibration_data.npz          # ST Edge AI calibration (key: 'input')
├── calibration_data.npy          # alternative format
├── calibration_data_flat.npy     # 2D fallback
├── calibration_labels.npy        # labels for verification
└── calibration_info.txt          # ST Edge AI instructions
```

```bash
python train_inceptiontime.py --epochs 50 --out inception_results
python train_inceptiontime.py --epochs 50 --n-cal 200 --out inception_results
```

---

## MS-RCFE + Ridge

### Overview

**MS-RCFE** is a lightweight feature extractor for time series. It uses **fixed** (untrained) random convolutional kernels to project the signal into a high-dimensional space, so that a linear model can classify it. The approach is inspired by the **ROCKET** paradigm (Dempster et al., 2020).

#### Architecture

The signal is processed by **9 parallel 1D convolutional branches** based on different time scales:

```
Signal x ∈ ℝ^T
      ↓
9 parallel Conv1D (fixed kernels, not trained)
  ├── kernel=3, dilation=1  →  max, mean  →  64 features
  ├── kernel=3, dilation=2  →  max, mean  →  64 features
  ├── kernel=3, dilation=4  →  max, mean  →  64 features
  ├── kernel=5, dilation=1  →  max, mean  →  64 features
  ├── kernel=5, dilation=2  →  max, mean  →  64 features
  ├── kernel=5, dilation=4  →  max, mean  →  64 features
  ├── kernel=9, dilation=1  →  max, mean  →  64 features
  ├── kernel=9, dilation=2  →  max, mean  →  64 features
  └── kernel=9, dilation=4  →  max, mean  →  64 features
      ↓
Concatenation  →  z ∈ ℝ^576
      ↓
Ridge classifier (linear)  →  ŷ ∈ {0, 1}
```

Two global statistics (**max** and **mean**) are extracted for each branch, producing a final vector of **576 features**. Classification is performed by a linear **Ridge classifier**, the only component optimized during training.

#### Main advantages

- **Fast training:** the convolutional part requires no backpropagation, analogously to an Extreme Learning Machine (Huang et al., 2006).
- **Multi-scale analysis:** identifies patterns at different frequencies and time resolutions, consistently with the MultiRocket approach (Tan et al., 2022).
- **Embedded design:** fixed weights drastically reduce memory usage, making the model compatible with the whole STM32 range.

### Training and export

```bash
python train_msrcfe.py --train train.npz --test test.npz --out results_msrcfe
python export_msrcfe.py msrcfe_bundle.pkl arc_dataset_train.npz --out export_msrcfe
```

**Output files:**

```text
export_msrcfe/
├── msrcfe.onnx               # feature extractor (fixed kernels) for ST Edge AI
├── ridge_weights.h           # Ridge classifier in C
└── calibration_msrcfe.npz    # INT8 calibration dataset
```

> **Note:** only the feature extractor (fixed kernels) is exported. The Ridge classifier is implemented in C through `ridge_weights.h`. Embedded inference consists of three steps: (1) applying the 9 fixed kernels to the raw signal, (2) computing the max and mean of each kernel, (3) a dot product with the Ridge weights.

#### Branch pruning and optimization studies

To maximize computational efficiency on very low-power microcontrollers, the folder includes a suite of analyses for reducing the convolutional branches:

- **`msrcfe_prune.py`:** systematic ablation of the 9 branches and feature ranking to identify minimal configurations.
- **`msrcfe_combinations.py`:** exhaustive sweep over the 511 possible combinations to determine the Pareto frontier (accuracy versus multiply-accumulate operations).
- **`msrcfe_reductions.py`:** ONNX and C header export for the reduced variants with 3 branches (`b3`, -82.4% MACs) and 4 branches (`b4`, -72.5% MACs), saved in `reduction_results/`.
- **`msrcfe_seed.py`:** statistical robustness test varying the seed of the random convolutional weights.

---

## MultiRocket and variants (comparative analysis)

`train_arc_compare.py` trains and compares three pipelines in a single run:

| Pipeline | Description |
|---|---|
| MultiRocket + Ridge | Feature extraction and linear classifier |
| MultiRocketHydra + Ridge | Hydra kernels with internal Ridge |
| MultiRocket + PCA + ArcNet | Feature extraction, PCA and neural network |

```bash
python train_arc_compare.py train.npz test.npz
python train_arc_compare.py train.npz test.npz --models ridge arcnet
python train_arc_compare.py train.npz test.npz --pca-components 128 --epochs 40
```

**Output files:**

```text
results/
├── ridge/
│   ├── bundle.pkl               model, transformer and scaler
│   └── config.json              metrics and estimated resources
├── hydra/
│   ├── bundle.pkl               full model
│   └── config.json              metrics
├── arcnet/
│   ├── bundle.pkl               transformer, scaler, PCA and model state_dict
│   └── config.json              metrics and estimated resources
├── comparison.json
├── comparison_metrics.csv
├── comparison_confusion.csv
├── comparison_resources.csv     resources per component
├── comparison_report.txt
└── comparison_plots.png         ROC, PR, confusion matrix, metrics
```

> **STM32 deployment note:** all three pipelines use MultiRocket for preprocessing. MultiRocket has no automatic C/ONNX export, so none of them can be deployed on STM32 without manually reimplementing the kernels in C. They were used exclusively for offline comparative analysis.

---

# 4. ONNX Export and Quantization

After training, the models are exported and quantized for embedded deployment.

| Model | ONNX export | INT8 quantization |
|---|---|---|
| InceptionTime | Yes (automatic during training) | Yes (via ST Edge AI) |
| MS-RCFE + Ridge | Yes (`export_msrcfe.py`, feature extractor only) | Yes (via ST Edge AI) |
| MultiRocket and variants | No | Not applicable |

---

# 5. Deployment with ST Edge AI

The ONNX models are validated with ST Edge AI Core / Developer Cloud.

For InceptionTime and MS-RCFE the flow is:

1. Import the `.onnx` model into ST Edge AI.
2. Load the `.npz` calibration dataset (key: `input`).
3. Select INT8 per-channel quantization.
4. Run the quantization to generate `model_int8.onnx`.
5. Load the quantized model into the STM32CubeIDE project.
6. Run inference on the test set and compare with FP32.

---

# 6. Causal Normalization and UL 1699B Compliance

To evaluate model robustness in realistic operating scenarios without prior knowledge of the signal, the folder `script/causal_current_norm/` implements:

- **Real-time causal normalization (`causal_norm.py`):** replaces the static nominal value with a moving estimate of $I_{nom}(t)$ computed only from past samples (circular buffer with hysteresis and anti-ratchet gating logic), showing that the models retain high performance under scale variations.
- **Intervention delay measurement (`detection_delay.py`):** simulates sliding-window processing on the test set and measures the time between arc onset and the positive decision confirmed over $K$ consecutive windows ($K=1, 2, 3$), verifying compliance with the **UL 1699B** time constraint ($t < 2.5\text{ s}$).

---

# 7. Robustness Analysis and Voltage-Based Ground Truth

To verify the reliability of the ground-truth labels and analyze model behavior with respect to the definition of the arc onset instant ($t_1$), the folder `script/voltage_label_creation/` includes:

- **Labeling parameter sensitivity (`label_sensitivity.py`):** evaluates the impact of varying threshold and confirmation-time criteria on pre-arc/arc separation, with 95% bootstrap confidence intervals.
- **Validation with voltage-based onset (`voltage_label_evaluation.py`):** compares the onset derived from the voltage drop of the generator/string ($t_{1,V}$) with the current-based one, stratifying model performance by current intensity band.
- **Parametric robustness (`voltage_label_robustness.py`):** sweeps voltage thresholds and confirmation windows, comparing the models with pure level-threshold baselines and highlighting ambiguous windows.
- **Delay diagnostics (`delay_analysis.py`):** analyzes, window by window, the edge cases with larger detection delays, jointly examining current, voltage and classifier scores.

---

# 8. Arc Energy Analysis

To assess the thermal and fire-ignition risk according to UL 1699B, the folder `script/energy_analysis/` includes:

- **Dissipated energy computation (`arc_energy.py`):** integrates the instantaneous power $E = \int_{t_1}^{t_1+T_{int}} v_{arc}(t) \cdot i(t) \, dt$ over a grid of synthetic extinction times, with compensation for the series resistance of the circuit.
- **Coupling with inference models:** combines the real detection delays (at confidence threshold $K$) of the InceptionTime and MS-RCFE models (FP32 and INT8) with the mechanical opening time of the relay (actuation), quantifying the effective energy before extinction.
- **Voltage-based onset support:** allows the energy integration to start from the transient instant detected on the voltage ($t_{1,V}$), capturing the dissipated energy also in cases of asymmetric or slow onset.

---

# 9. Variable-Load Analysis and Spectral Robustness

In real photovoltaic systems, irradiance variations and inverter switching introduce current fluctuations and transients that can confuse detectors based on simple amplitude thresholds. The folder `script/variable_load_analysis/` systematically evaluates the ability of the models to discriminate arcs under dynamic conditions:

- **Impact of load variations (`variable_load.py`):** evaluates detection accuracy in the presence of load steps and switching transients, verifying the separability between physiological plant dynamics and high-frequency perturbations induced by the arc.
- **Load scenario fusion (`load_fusion.py`):** evaluates classifier performance by combining heterogeneous recordings and different load profiles, verifying cross-load generalization.
- **Level-independent detection (`level_free_detection.py`):** isolates the AC components and high-frequency morphological features (removing the DC offset), assessing whether the models learn the intrinsic signature of the arc discharge rather than relying on the drop in mean current level.
- **Spectral analysis and decomposition (`mean_spectrum.py`):** computes the mean power spectral density (PSD) by time band during the different phases (pre-arc, onset transient, steady-state arc), quantifying the spectral energy in the operating bands.
- **Multichannel framework (`multichannel_analysis.py`):** processes and extracts joint features from multiple channels (current, voltage, instantaneous power) to improve intervention timeliness and reduce false alarms (FPR).
- **Waveform and channel visualization (`channel_plot.py`):** generates diagnostic plots of the time-domain waveforms and of the associated synchronized signals.

---

# Experimental Results

The results below report classification performance on the test set, hardware benchmarks measured on the target microcontrollers and regulatory compliance checks.

---

## 1. Classification Results on the Test Set

Evaluation on **7,161 test windows** (4,520 arc, 2,641 pre-arc) from **215 independent physical experiments**. The 95% confidence intervals are obtained by **bootstrap resampling with 2,000 replicates** at the experiment level.

| Model | Precision | Detection Rate (DR) [%] | False Positive Rate (FPR) [%] | FN | FP | Accuracy |
|---|---|---|---|---|---|---|
| **InceptionTime** | FP32 | 99.7% [99.4, 100.0] | 0.61% [0.23, 1.09] | 13 | 16 | **99.60%** |
| **InceptionTime** | INT8 | 98.5% [97.4, 99.3] | 0.15% [0.00, 0.47] | 70 | 4 | **98.97%** |
| **MS-RCFE + Ridge** | FP32 | 99.5% [99.0, 99.9] | 0.11% [0.00, 0.31] | 23 | 3 | **99.64%** |
| **MS-RCFE + Ridge** | INT8 | 99.2% [98.6, 99.6] | 0.11% [0.00, 0.31] | 38 | 3 | **99.43%** |

> **Impact of INT8 quantization:** post-training quantization reduces accuracy by only 0.63 percentage points for InceptionTime and by 0.21 points for MS-RCFE. The drop translates mainly into missed detections (FN) rather than false alarms (FP), preserving a very low FPR ($\le 0.15\%$).

---

## 2. STM32 Hardware Benchmark: Measured Inference Latency

Inference latency measured on the physical target hardware for a single 100 ms window (1,000 samples at 10 kHz):

- **STM32N6570-DK:** microcontroller with a **Neural-ART NPU** hardware accelerator.
- **STM32H7S78-DK:** general-purpose microcontroller based on an **ARM Cortex-M7** (600 MHz) without accelerator.

| Board / Platform | Acceleration | Model | FP32 [ms] | INT8 [ms] | INT8 throughput |
|---|---|---|---|---|---|
| **STM32N6570-DK** | Neural-ART NPU | InceptionTime | 6218.00 ms | **19.73 ms** | **50.68 inf/s** |
| **STM32H7S78-DK** | Cortex-M7 (CPU) | InceptionTime | 6997.67 ms | 2370.64 ms | 0.42 inf/s |
| **STM32H7S78-DK** | Cortex-M7 (CPU) | MS-RCFE + Ridge | 84.75 ms | **69.77 ms** | **14.33 inf/s** |

### Embedded Memory Resources

| Parameter / Resource | InceptionTime | MS-RCFE + Ridge |
|---|---|---|
| **Input** | Raw window (100 ms, 1,000 samples) | Raw window (100 ms, 1,000 samples) |
| **Feature extractor** | End-to-end trained convolutions | $9 \times 32$ fixed random kernels (576 features) |
| **Classifier** | Softmax | Ridge ($\alpha = 1$, weights in C) |
| **MAC operations** | — | **1.632 M** |
| **Ideal target hardware** | STM32N6 (Neural-ART NPU) | STM32H7S78 (Cortex-M7 without NPU) |
| **Precision** | FP32, INT8 | FP32, INT8 |
| **Runtime RAM (INT8)** | **2.15 MB** | **≈ 141 KiB** (141.11 KB) |
| **Total Flash (INT8)** | 399 KB (INT8 weights) | **30.81 KB** (23.56 KB FP32) |

> **Key deployment considerations:**
> 1. **InceptionTime** requires the NPU: on a Cortex-M55 without NPU it takes 6.22 s, whereas the Neural-ART accelerator reduces latency to **19.73 ms** ($\sim 315\times$ speed-up), compatible with the 50 ms overlap step at the cost of 2.15 MB of internal RAM.
> 2. **MS-RCFE + Ridge** requires only **141 KiB of RAM** and **30.8 KB of Flash**. Running on the Cortex-M7 without any hardware accelerator, it completes inference in **69.77 ms** in INT8 (84.75 ms in FP32).

---

## 3. UL 1699B Safety Limit Verification

Sliding-window simulation performed on all **442 test experiments containing an arc**. The delay is measured from the physical onset instant $t_1$ to the completion of the window that confirms $K$ consecutive positive predictions (excluding the mechanical actuation time). The energy is computed by integrating $E = \int v(t) i(t) dt$.

### Indicative Comparison with UL 1699B Limits

| Measured quantity | Median | Worst case (max) | UL 1699B limit | Status |
|---|---|---|---|---|
| **Detection delay ($K = 1$)** | **117 – 143 ms** | **1940 ms** (1.94 s) | — | Compliant |
| **Delay with 50 ms actuation ($K = 3$)** | **0.24 – 0.29 s** | **2.09 s** (2090 ms) | **2.50 s** | **Compliant** |
| **Arc energy ($K = 1$, 0 ms actuation)** | **9.3 – 13.6 J** | **267.7 J** | **750.0 J** | **Compliant** |

### Energy Dissipated Until Detection ($K = 1$, 0 ms Actuation)

| Model | Precision | Median [J] | 95th percentile [J] | Absolute maximum [J] |
|---|---|---|---|---|
| **InceptionTime** | FP32 | 9.3 J | 19.1 J | 36.9 J |
| **InceptionTime** | INT8 | 13.6 J | 53.5 J | 267.7 J |
| **MS-RCFE + Ridge** | FP32 | 12.8 J | 30.2 J | 93.5 J |
| **MS-RCFE + Ridge** | INT8 | 12.9 J | 35.2 J | 251.5 J |

---

## 4. Sensitivity to Labeling Parameters

Robustness of the trained models (without retraining) with respect to variations in the ground-truth definition criteria (current amplitude threshold on $I_{nom}$ and temporal confirmation window):

| Label configuration | Inception FP32 (DR / FPR %) | Inception INT8 (DR / FPR %) | MS-RCFE FP32 (DR / FPR %) | MS-RCFE INT8 (DR / FPR %) |
|---|---|---|---|---|
| **Base (threshold 0.95, confirmation 300 ms)** | **99.7 / 0.6** | **98.5 / 0.2** | **99.5 / 0.1** | **99.2 / 0.1** |
| Threshold 0.90 | 100 / 47.3 | 100 / 40.8 | 100 / 44.0 | 100 / 43.2 |
| Threshold 0.93 | 100 / 21.0 | 100 / 9.2 | 100 / 14.6 | 100 / 13.5 |
| Threshold 0.97 | 96.8 / 0.1 | 93.2 / 0.2 | 95.2 / 0.1 | 94.9 / 0.1 |
| Threshold 0.98 | 96.0 / 0.1 | 92.6 / 0.2 | 94.6 / 0.1 | 94.3 / 0.1 |
| Confirmation time 100 ms | 99.4 / 0.1 | 97.9 / 0.0 | 99.0 / 0.0 | 98.7 / 0.0 |
| Confirmation time 1000 ms | 100 / 1.6 | 98.9 / 0.3 | 99.9 / 0.6 | 99.6 / 0.4 |

> The test shows that the classifiers learn the decision boundary on which they were supervised, which makes it important to analyze the physical separability of the high-frequency features with respect to the DC component alone.

---

## 5. MS-RCFE Branch Ablation and Pruning

Reduction study of the 9 convolutional branches (evaluated over 10 random extractors with independent seeds, 9,000 training windows) to identify Pareto-optimal configurations with very low latency:

| Branch subset | DR [%] | Balanced accuracy [%] | MAC operations | INT8 latency (STM32H7S78) | INT8 latency (STM32N6570) |
|---|---|---|---|---|---|
| **All 9 branches (full)** | **99.32 ± 0.23%** | **99.59 ± 0.10%** | **1.632 M** (100%) | **69.77 ms** | **88.81 ms** |
| 6 branches ($k = 3, 5$) | 98.92 ± 0.66% | 99.41 ± 0.32% | 768 k (47.1%) | — | — |
| **4 branches (`b4`)** | 98.52 ± 1.00% | 99.22 ± 0.49% | 448 k (27.5%) | **26.46 ms** | **32.46 ms** |
| **3 branches (`b3`)** | 98.06 ± 1.05% | 98.99 ± 0.52% | 288 k (17.6%) | **17.78 ms** | **21.13 ms** |
| 1 branch ($k = 3, d = 2$) | 95.29 ± 2.06% | 97.61 ± 1.03% | 96 k (5.9%) | — | — |
| Optimum selected via CV | 99.30 ± 0.22% | 99.58 ± 0.10% | 22% – 67% | — | — |

> **Practical impact:** the reduced variants with 3 branches (`b3`: $k=3$, $d=1,2,4$) and 4 branches (`b4`: $k=3$ with $d=1,2,4$ plus $k=5$ with $d=2$) reduce the MACs by 82.4% and 72.5%, lowering latency on the Cortex-M7 to **17.78 ms** and **26.46 ms**. This allows MS-RCFE to run well below the 50 ms time constraint even on conventional microcontrollers without accelerators.

---

## 6. MultiRocket Baseline Comparison (Offline Only)

| Pipeline | Accuracy | Balanced accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| MultiRocket + Ridge | 99.54% | 99.52% | 99.63% | 99.93% |
| MultiRocketHydra + Ridge | **99.65%** | **99.66%** | **99.72%** | 99.66% |
| MultiRocket + PCA + ArcNet | 99.55% | 99.52% | 99.65% | **99.99%** |

> The resources of MultiRocketHydra cannot be estimated: the internal Hydra kernels are not accessible as arrays.

---

# Conclusions

The project evaluated two main families of models for DC arc fault detection, with joint requirements of high accuracy and real-time embedded compatibility.

**MS-RCFE + Ridge** is the preferred solution for low-cost microcontrollers without an accelerator (for example, STM32H7 with Cortex-M7). It combines an accuracy of **99.43%** even after INT8 quantization, a minimal hardware footprint (**30.8 KB Flash, 141 KiB RAM**) and an inference time of **69.77 ms** in INT8 (84.75 ms in FP32). With the pruned variants of 3 branches (`b3`) and 4 branches (`b4`), latency is further reduced to **17.78 ms** and **26.46 ms**, enabling continuous inference with a 50 ms overlap step.

**InceptionTime** is the high-performance choice when a dedicated hardware accelerator is available (**Neural-ART NPU on the STM32N6570-DK**). The NPU brings latency to **19.73 ms** (a **~315×** speed-up over the 6218 ms of the Cortex-M55 CPU alone) with an accuracy of **98.97%**, requiring 2.15 MB of internal RAM.

Both solutions meet the **UL 1699B** requirements, with median delays of 117–143 ms ($< 2.5\text{ s}$) and energy dissipated before disconnection below 267.7 J ($< 750\text{ J}$).

---

# Authors and Affiliation

- **Lorenzo Meloccaro**, Università Politecnica delle Marche
- **Yassir Flavio Suarez Sanchez**, Università Politecnica delle Marche

---

# License

The original dataset belongs to the respective authors of the project published on IEEE DataPort.
The repository code is distributed under the license specified in the project.
