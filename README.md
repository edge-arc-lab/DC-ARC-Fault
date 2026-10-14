# DC-ARC-Fault ⚡

DC arc faults are a critical risk in photovoltaic (PV) plants and one of the main causes of fires. Reliable detection is difficult because the signals are noisy and non-stationary.

This work addresses the problem with reference to **UL 1699B**, developing a deep-learning-based detection system designed not only for high accuracy but also to be deployable in real time on **STM32** embedded devices.

The project covers:

- time-series preprocessing and labelling
- **data leakage** control
- training of **deep learning** and **feature-based** models
- **ONNX** export
- **INT8** quantisation
- edge-oriented benchmarks on **STM32**
- extended analyses: dependence on the current level, detection delay, arc energy, causal normalisation, variable load, label-parameter sensitivity

---

# 🧠 Project Goal

The goal is to develop a classification system able to identify DC arc faults in PV plants, while jointly optimising:

- model accuracy
- robustness against **data leakage**
- compatibility with embedded systems
- **real-time** inference on edge hardware

The pipeline is designed to be reproducible and edge-ready. The reported results come from a single training run per model (no seed control, no independent validation set); the limitations are discussed in [Limitations](#-limitations).

---

# 📁 Project Structure

```text
DC-ARC-Fault/
│
├── dataset/
│   ├── dataset/                          # Original dataset (Kaggle / IEEE DataPort)
│   ├── dataset_new/                      # Final preprocessed dataset
│   ├── estrazione e sottocampionamento/
│   │   └── mat_tools.py                  # .mat extraction and 25x downsampling
│   ├── build_dataset_new.py              # Sliding window, labelling, normalisation
│   ├── split_dataset.py                  # Physical train/test split
│   └── dataset_leakage_check.py          # Data leakage verification
│
├── script/
│   ├── training/
│   │   ├── inception_time/
│   │   │   ├── results/
│   │   │   └── train_inceptiontime.py    # Training + ONNX export + calibration
│   │   ├── msrcfe/
│   │   │   ├── results/
│   │   │   ├── export_msrcfe/
│   │   │   ├── train_msrcfe.py           # MS-RCFE + Ridge training
│   │   │   └── export_msrcfe.py          # ONNX export + calibration
│   │   └── multirockethydra/
│   │       ├── results/
│   │       └── train_arc_compare.py      # Comparative training (Ridge / Hydra / ArcNet)
│   │
│   ├── modelli_quantizzati/              # INT8 models and comparison scripts
│   │   ├── Inception/
│   │   └── msrcfe/
│   │
│   ├── test_set_analysis.py              # Bootstrap, elementary baselines, error analysis
│   ├── labels_analysis/                  # Label-parameter sensitivity
│   │   ├── lab_analysis_1.0.py
│   │   └── lab_analysis_2.0.py
│   ├── casual current I_norm/            # Causal I_nom and detection delay
│   │   ├── norm_causale_v1.py
│   │   ├── norm_causale_v2.py
│   │   └── ritardo_rilevazione_v1.py
│   ├── energy_analysis/
│   │   └── energia_arco_v2.py            # Arc energy estimate
│   └── variable load analysis/
│       └── var_load_analysis.py          # Variable-load analysis
│
├── results_data/                         # Results of the extended analyses
├── pipeline.png / pipeline_v2_en.pdf     # Pipeline diagrams
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the repository

```bash
git clone https://github.com/yassirsuarez/DC-ARC-Fault.git
cd DC-ARC-Fault
```

## 2️⃣ Create a virtual environment

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

## 3️⃣ Install the dependencies

```bash
pip install -r requirements.txt
```

---

# 📚 Dataset

The project uses the public dataset:

**Photovoltaic (PV) DC Arc-Fault Library**
[IEEE DataPort](https://ieee-dataport.org/open-access/photovoltaic-pv-dc-arc-library)

From the original dataset, **current**, **voltage** and **power** are extracted (only the current is used by the models).

A preprocessed version is also available on Kaggle:
[Kaggle – DC Arc Fault Dataset](https://www.kaggle.com/datasets/yassirsuarez/dc-arc-fault)

Download the dataset and place it in `dataset/dataset/`. Expected structure:

```text
dataset/
└── dataset/
    ├── E001_0050V_00.50A_G001
    ├── E001_0050V_00.50A_G002
    └── ...
```

---

# 🔄 Project Pipeline

![Pipeline](pipeline_v2_en.png)

The vector version of the diagram is `pipeline_v2_en.pdf`. Stages 1–5 are the development and validation flow; stage 6 collects the analyses run on the same test set without retraining the models.

---

# 1️⃣ Data management, preprocessing and physical split

`mat_tools.py` reduces the original files once: it keeps the `.mat` files, extracts current, voltage and power and downsamples by a factor of 25 (250 kHz → 10 kHz).

The signals are then processed by:

- **Sliding window**: 100 ms windows (1000 samples), 50 ms step, at most 10 windows per class per file
- **Labelling**: the arc onset `t1` is the first sample where the current drops below 0.95 × the nominal current for at least 300 ms; windows before `t1` are *pre-arc*, windows after are *arc*
- **Normalisation** by the nominal current `I_nom`, estimated as the median of the first 0.2 s of each recording (minimum accepted: 0.10 A)

```bash
python dataset/build_dataset_new.py
python dataset/split_dataset.py
```

The physical split (80% train / 20% test) is done **per experiment**, **before** training, to avoid temporal leakage between windows of the same recording. Output: `.npz` for training, `.csv` for analysis. The test set has 7161 windows (4520 arc, 2641 pre-arc) from 215 experiments.

> **Known limitation:** labels are defined on the normalised current level, and `I_nom` is not available in real time on a plant. See [Limitations](#-limitations).

---

# 2️⃣ Data leakage verification

Verification that there is no data leakage between training and test sets:

- search for identical windows (exact comparison) and near-duplicates (on a subset)
- check that no experiment appears in both sets (by file name)
- shuffled-label test on elementary features, expected to give a balanced accuracy close to chance

```bash
python dataset/dataset_leakage_check.py
```

---

# 3️⃣ Training & Feature Extraction

The folder `script/training/` contains the training pipelines for three architectures.

## Models

| Model | Literature name | Type | STM32 deployment |
|---|---|---|---|
| InceptionTime | InceptionTime | End-to-end deep learning (trained temporal CNN) | ✅ ONNX → ST Edge AI |
| MS-RCFE + Ridge | MS-RCFE | Multi-scale fixed-kernel ROCKET + linear classifier | ✅ ONNX → ST Edge AI |
| MultiRocket + variants | ROCKET / MultiRocket | Feature-based (random kernels + classical ML) | ⚠️ Experimental only |

---

## 🧠 InceptionTime

`train_inceptiontime.py` handles in a single script:

- training with tsai + PyTorch on GPU
- UL 1699B-style metrics (detection rate, false positive rate)
- multi-threshold analysis
- export of `inceptiontime.onnx` (dynamic shape)
- generation of the calibration dataset for INT8 quantisation

**Output:**

```text
risultati_inception/
├── inceptiontime.onnx            ← for ST Edge AI
├── inceptiontime_training.png    ← loss/accuracy curves
├── results_inceptiontime.png     ← confusion matrix, ROC, PR, score distribution
├── inceptiontime_report.txt      ← full metrics
├── calibration_data.npz          ← ST Edge AI calibration (key: 'input')
├── calibration_data.npy          ← alternative
├── calibration_data_flat.npy     ← 2D fallback
├── calibration_labels.npy        ← labels for verification
└── calibration_info.txt          ← ST Edge AI instructions
```

```bash
python train_inceptiontime.py --epochs 50 --out risultati_inception
python train_inceptiontime.py --epochs 50 --n-cal 200 --out risultati_inception
```

---

## 🧠 MS-RCFE + Ridge

### What is MS-RCFE

**MS-RCFE** is an ultra-light time-series feature extractor. It uses **fixed** (untrained) random convolutional kernels to project the signal into a high-dimensional space, so that a linear model can classify it. The approach is inspired by the **ROCKET** paradigm (Dempster et al., 2020).

#### Architecture

The signal goes through **9 parallel 1D convolutional branches** at different temporal scales:

```
Signal x ∈ ℝ^T
      ↓
9 parallel Conv1D (fixed kernels, not trained)
  ├── kernel=3, dilation=1  →  max, mean  →  64 feat
  ├── kernel=3, dilation=2  →  max, mean  →  64 feat
  ├── kernel=3, dilation=4  →  max, mean  →  64 feat
  ├── kernel=5, dilation=1  →  max, mean  →  64 feat
  ├── kernel=5, dilation=2  →  max, mean  →  64 feat
  ├── kernel=5, dilation=4  →  max, mean  →  64 feat
  ├── kernel=9, dilation=1  →  max, mean  →  64 feat
  ├── kernel=9, dilation=2  →  max, mean  →  64 feat
  └── kernel=9, dilation=4  →  max, mean  →  64 feat
      ↓
Concatenation  →  z ∈ ℝ^576
      ↓
Ridge Classifier (linear)  →  ŷ ∈ {0, 1}
```

Two global statistics (**max** and **mean**) are extracted from each branch, giving a final vector of **576 features**. Classification is done by a linear **Ridge Classifier**, the only component optimised during training.

#### Main advantages

- **Very fast training:** no backpropagation for the convolutional part, as in an Extreme Learning Machine (Huang et al., 2006).
- **Multi-scale analysis:** captures patterns at different frequencies and temporal resolutions, in line with MultiRocket (Tan et al., 2022).
- **Embedded-oriented design:** fixed weights greatly reduce the memory footprint.

> The 9 kernels with length 9 account for about 52.9% of the multiply-accumulate operations of the extractor. Reducing the branches (retraining only the Ridge) is left as future work.

### Training and export

```bash
python train_msrcfe.py --train train.npz --test test.npz --out results_msrcfe
python export_msrcfe.py msrcfe_bundle.pkl arc_dataset_train.npz --out export_msrcfe
```

**Output:**

```text
export_msrcfe/
├── msrcfe.onnx               ← feature extractor (fixed kernels) for ST Edge AI
├── ridge_weights.h           ← Ridge classifier in C
└── calibration_msrcfe.npz   ← INT8 calibration dataset
```

> **Note:** only the feature extractor (fixed kernels) is exported. The Ridge classifier is implemented in C through `ridge_weights.h`. Embedded inference has three steps: (1) apply the 9 fixed kernels to the raw signal, (2) compute max and mean for each kernel, (3) dot product with the Ridge weights.

---

## 🚀 MultiRocket + variants (comparative analysis)

`train_arc_compare.py` trains and compares three pipelines in a single run:

| Pipeline | Description |
|---|---|
| MultiRocket + Ridge | Feature extraction + linear classifier |
| MultiRocketHydra + Ridge | Hydra kernels + internal Ridge |
| MultiRocket + PCA + ArcNet | Feature extraction + PCA + neural network |

```bash
python train_arc_compare.py train.npz test.npz
python train_arc_compare.py train.npz test.npz --models ridge arcnet
python train_arc_compare.py train.npz test.npz --pca-components 128 --epochs 40
```

**Output:**

```text
results/
├── ridge/
│   ├── bundle.pkl               model + transformer + scaler
│   └── config.json              metrics + estimated resources
├── hydra/
│   ├── bundle.pkl               full model
│   └── config.json              metrics
├── arcnet/
│   ├── bundle.pkl               transformer + scaler + pca + model state_dict
│   └── config.json              metrics + estimated resources
├── comparison.json
├── comparison_metrics.csv
├── comparison_confusion.csv
├── comparison_resources.csv     per-component resources
├── comparison_report.txt
└── comparison_plots.png         ROC, PR, confusion matrix, metrics
```

> **STM32 deployment note:** all three pipelines use MultiRocket as preprocessing. MultiRocket has no automatic C/ONNX export, so none of them can be deployed on STM32 without manually reimplementing the kernels in C. They were used only for offline comparison.

---

# 4️⃣ ONNX export & quantisation

After training, the models are exported and quantised for embedded deployment.

| Model | ONNX export | INT8 quantisation |
|---|---|---|
| InceptionTime | ✅ automatic in training | ✅ via ST Edge AI |
| MS-RCFE + Ridge | ✅ `export_msrcfe.py` (feature extractor only) | ✅ via ST Edge AI |
| MultiRocket + variants | ❌ not available | ❌ not applicable |

---

# 5️⃣ Deployment check on ST Edge AI

The ONNX models are validated through ST Edge AI Core / Developer Cloud.

For InceptionTime and MS-RCFE the flow is:

1. Import the `.onnx` model into ST Edge AI
2. Load the `.npz` calibration dataset (key: `input`)
3. Select INT8 per-channel quantisation
4. Run the quantisation → generates `model_int8.onnx`
5. Run the benchmark on the target board (Developer Cloud)

> The classification metrics (FP32 vs INT8) were computed with **onnxruntime on a CPU**, on the independent test set; the benchmarks on the boards measure latency and memory, not accuracy.

---

# 6️⃣ Extended analyses (no retraining)

These analyses use the models already trained and the same test set. Scripts are in `script/` and results in `results_data/`.

| Analysis | Script | Main finding |
|---|---|---|
| Level bands and elementary baselines | `test_set_analysis.py`, `sottoinsiemi_livello_v1.py` / `v2.py` (to be added to the repository) | A logistic regression on the window mean, min and max reaches about 99.7%: accuracy mostly measures the ability to reproduce a threshold on the current level |
| Detection delay (sliding window) | `casual current I_norm/ritardo_rilevazione_v1.py` | Median delay 117–143 ms (K = 1); maximum 2040 ms (K = 3, InceptionTime INT8) |
| Arc energy | `energy_analysis/energia_arco_v2.py` | Median 9–31 J; maximum 267.7 J (upper bound, series resistance set to 0) |
| Causal `I_nom` and variable load | `casual current I_norm/norm_causale_v2.py`, `variable load analysis/` | A moving median over the last W seconds with a gate on the level keeps the detection rate within 0.1 points of the static one under stable load; ramps and steps degrade it |
| Label-parameter sensitivity | `labels_analysis/lab_analysis_2.0.py` | The relative threshold dominates (0.93/0.90: false-alarm rate 9–47%; 0.97/0.98: detection rate 92.6–97%); confirmation time has a limited effect; minimum current has none |

**UL 1699B time and energy criteria** (Table 29, as given in the project specification: 2.5 s to detect and interrupt the arc, 750 J total arc energy). The simulated maxima (detection delay 2040 ms plus up to 50 ms of actuation, and 267.7 J) are below the limits, with a narrow margin on delay in the tails of the INT8 versions. The comparison is indicative: the delay is referred to `t1` defined from the current level, the energy is an upper bound, and the inference and actuation times on the device are not included, so it is not a conformity proof. The limits have not been verified on the text of the standard.

---

# 📊 Results

## FP32 vs INT8 accuracy (independent test set, onnxruntime on CPU)

| Model | FP32 accuracy | INT8 accuracy |
|---|---|---|
| InceptionTime | 99.60% | 98.97% |
| MS-RCFE + Ridge | 99.64% | 99.43% |

Both models stay above 98.9% after INT8 quantisation. These figures come from a single training run per model, and the difference between the two models cannot be separated from the effect of the initialisation. Because labels depend on the current level, the accuracies should be read together with the level-only baselines of stage 6.

## MultiRocket comparison (offline only)

| Pipeline | Accuracy | Balanced Acc | F1 | ROC-AUC |
|---|---|---|---|---|
| MultiRocket + Ridge | 99.54% | 99.52% | 99.63% | 99.93% |
| MultiRocketHydra + Ridge | **99.65%** | **99.66%** | **99.72%** | 99.66% |
| MultiRocket + PCA + ArcNet | 99.55% | 99.52% | 99.65% | **99.99%** |

---

## ⚡ STM32 benchmark — InceptionTime

Tested on **STM32N6570-DK** (Cortex-M55 at 800 MHz) with the **Neural-ART NPU**, comparing FP32 (CPU) and INT8 (NPU). Three sessions per model, ST Edge AI Developer Cloud.

| Metric | FP32 (CPU) | INT8 (NPU) |
|---|---|---|
| Inference time (mean of 3 sessions) | 6218 ms (6178.5 / 6247.0 / 6228.7) | 19.73 ms (19.715 / 19.735 / 19.729) |
| Cycles | 4.98 billion | 15.78 million |
| Throughput | about 0.16 inf/s | about 50.7 inf/s |
| Internal RAM | 4.74 MB | 2.15 MB |

The INT8/NPU version is about **315×** faster than FP32 on CPU. 19.73 ms is below the 50 ms window step (about 39% load) and measures inference only. On the N6 the Developer Cloud does not allow choosing NPU or CPU for INT8, so the effect of quantisation alone was measured on the H7S78 (below). The report does not include flash, weights and total RAM, so those figures are not given here. MS-RCFE was not measured on the N6 (it cannot be measured directly with ST Edge AI on that board).

**Quantisation alone, on STM32H7S78 (Cortex-M7 at 600 MHz, no NPU):** InceptionTime INT8 2370.64 ms vs FP32 6997.67 ms (about ×2.95); weights 394,376 B vs 1,554,056 B; total RAM 950,480 B vs 1,942,968 B. This is an indication, not transferable to the N6 (different CPU and external memory).

The ONNX file hashes match the ones reported in the ST Edge AI CSV (INT8 `5ff6de27…`, FP32 `c9d6d54e…`).

---

## ⚡ STM32 benchmark — MS-RCFE + Ridge

Tested on **STM32H7S78-DK** (Cortex-M7), comparing the full pipeline (feature extractor + Ridge) in FP32 vs INT8, measured **directly on the real board** through ST Edge AI Developer Cloud.

| Metric | FP32 | INT8 | Δ |
|---|---|---|---|
| Inference time | 84.75 ms | 70.04 ms | ↓ ~1.2× |
| Throughput | 11.80 inf/s | 14.28 inf/s | ↑ ~1.2× |
| Total RAM | 135.61 KB | 141.11 KB | +4% |
| Total Flash | 23.56 KB | 30.81 KB | +30% |

**Preliminary estimate of the relative weight of the components (host CPU benchmark, not the board):**

Before deployment, a local benchmark simulating inference with `onnxruntime` on the development PC CPU (script `misura_risorse.py`) was run to estimate the relative weight of the CNN and the Ridge within the pipeline:

| Component (host CPU, onnxruntime) | FP32 | INT8 |
|---|---|---|
| Feature extractor (ONNX) | 0.333 ms | 0.684 ms |
| Ridge Classifier (C, simulated) | 0.006 ms | 0.006 ms |

> **Warning:** these values are **neither comparable nor additive** with the end-to-end latency measured on the board (84.75 / 70.04 ms): they come from two different computing platforms. The host CPU is hundreds of times more powerful than the Cortex-M7. The CPU benchmark did confirm, proportionally, that the Ridge Classifier is computationally negligible compared with the feature extractor. A real per-component breakdown measured on the board is not available.

Both configurations tested on real hardware are well above the 50 Hz real-time requirement. INT8 quantisation does not always improve latency on small models: here it introduces a slight overhead on the CNN but reduces total memory use.

> **Note on memory values:** the Flash and RAM values include the library runtime (about 9 KB overhead). The analytical estimate of the model weights alone (feature extractor + Ridge) is **14.7 KB Flash · 10.7 KB RAM**.

---

## 📊 Embedded resource estimate — MultiRocket variants

The MultiRocket pipelines were not deployed on STM32 (preprocessing not exportable to C). The table gives an analytical estimate based on the sizes of the trained models. The cost of the MultiRocket kernels (~24.4 KB Flash) is common to Ridge and ArcNet and is included.

| Pipeline | Internal flash | External flash | RAM |
|---|---|---|---|
| MultiRocket + Ridge | ~607 KB | — | ~194 KB |
| MultiRocket + PCA + ArcNet | ~680 KB | ~48.6 MB (QSPI) | ~195 KB |

> MultiRocketHydra cannot be estimated: the internal Hydra kernels are not accessible as arrays.

---

# ⚠️ Limitations

- **Labels depend on the current level.** `t1` is defined on the normalised current, so the models partly learn a threshold on the level (a three-number logistic baseline reaches about 99.7%). Performance changes substantially with the threshold (stage 6).
- **`I_nom` is estimated from the first 0.2 s** of each recording, which is not available in real time on a plant; a causal estimate is evaluated in stage 6 but the models were not retrained with it.
- **Single training run per model**, no seed control and no independent validation set; the difference between the models cannot be separated from the effect of the initialisation.
- **Classification metrics from onnxruntime on CPU**, not measured on the target; the INT8/NPU values for InceptionTime on the N6 are not backed by a saved full report (weights and total RAM missing).
- **UL 1699B:** the 2.5 s / 750 J criteria are compared only with simulated estimates; end-to-end interruption time is not measured. The DR ≥ 95% and FPR ≤ 5% criterion is adopted in this work and is not prescribed by the standard.

---

# 🏁 Conclusions and Recommendations

The project evaluated three families of models for DC arc-fault detection, with joint requirements of high accuracy and embedded compatibility.

**MS-RCFE + Ridge** is the recommended model for deployment on mid-range STM32 devices (e.g. STM32H7). It combines an accuracy above 99.4% after INT8 quantisation, a minimal hardware footprint (23.6 KB Flash, 135.6 KB RAM measured on hardware) and an INT8 latency of 70 ms, well within the 50 Hz real-time requirement. The pipeline can be exported through ST Edge AI without firmware changes.

**InceptionTime** is the right choice when a board with a dedicated NPU is available (e.g. STM32N6570-DK with Neural-ART). In that case the latency gain is about 315× compared with FP32, making continuous inference practical. On Cortex-M boards without an NPU the FP32 latency is prohibitive for real-time use.

**MultiRocket and variants** give the highest classification metrics (F1 up to 99.72% with Hydra) but cannot be deployed on STM32 without manually reimplementing the preprocessing in C. They were used only as an offline comparative baseline.

---

# 👥 Team

- **Lorenzo Meloccaro** — MSc, Università Politecnica delle Marche (UNIVPM)
- **Yassir Flavio Suarez Sanchez** — MSc, Università Politecnica delle Marche (UNIVPM)

---

# 📄 License

The original dataset belongs to the authors of the project published on IEEE DataPort.
The repository code is distributed under the licence specified in the project.
