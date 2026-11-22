# Edge AI for DC Arc Fault Detection

I guasti da arco elettrico in serie in corrente continua (DC series arc faults) rappresentano un rischio critico per gli impianti fotovoltaici e costituiscono una delle cause primarie di incendio. Questo lavoro affronta il problema nel pieno rispetto dei vincoli temporali ed energetici dello standard **UL 1699B** (e controparte internazionale **IEC 63027**), progettando, quantizzando e validando su hardware reale **STM32** due architetture complementari:

1. **InceptionTime:** rete convoluzionale temporale profonda ottimizzata per microcontrollori con NPU dedicata (**STM32N6570-DK** con Neural-ART NPU).
2. **MS-RCFE + Ridge:** estrattore multi-scala a 9 rami convoluzionali casuali fissi con classificatore Ridge lineare in C, ideato per microcontrollori general-purpose senza acceleratori (**STM32H7S78-DK** con Cortex-M7 a 600 MHz).

### Risultati Chiave
- **Detection Rate (DR):** tra **98.5%** e **99.7%** su tutte le varianti (FP32 e INT8).
- **False-Positive Rate (FPR):** tra **0.11%** e **0.61%** (zero falsi allarmi critici).
- **Latenza di inferenza reale misurata su hardware:**
  - **InceptionTime INT8:** **19.73 ms** su NPU Neural-ART (accelerazione **~315×** rispetto a 6218 ms su CPU).
  - **MS-RCFE INT8:** **69.77 ms** su Cortex-M7 puro (RAM: **≈ 141 KiB**, Flash: **30.8 KB**); scende a **17.78 ms** nella variante potata a 3 rami (`b3`).
- **Ritardo e bilancio energetico (UL 1699B):**
  - Ritardo mediano di rilevamento: **117 – 143 ms** (caso peggiore $1.94\text{ s} < 2.5\text{ s}$).
  - Energia dissipata fino all'estinzione: mediana **9.3 – 13.6 J**, massimo assoluto **267.7 J** ($< 750\text{ J}$).

---

# Obiettivo del Progetto

L'obiettivo è sviluppare un sistema di classificazione capace di identificare fault da arco elettrico DC in impianti fotovoltaici, ottimizzando contemporaneamente:

- accuratezza del modello
- robustezza rispetto al **data leakage**
- compatibilità con sistemi embedded
- inferenza **real-time** su hardware edge

L'intera pipeline è progettata per essere riproducibile, scalabile ed edge-ready.

---

# Struttura del Progetto

```text
DC-ARC-Fault/
│
├── dataset/
│   ├── dataset/                          # Dataset compresso a 10 kHz (da Kaggle)
│   ├── dataset_new/                      # Dataset finestrato finale e split train/test
│   │   ├── arc_dataset_new.npz           # Dataset completo finestrato (X, y)
│   │   ├── arc_dataset_meta_new.csv      # Metadati di tutte le finestre temporali
│   │   ├── arc_dataset_skipped_new.csv   # File scartati e motivazione dello scarto
│   │   ├── arc_dataset_train.npz         # Set di training (80% dei file, no leakage)
│   │   ├── arc_dataset_meta_train.csv    # Metadati del set di training
│   │   ├── arc_dataset_test.npz          # Set di test (20% dei file, no leakage)
│   │   ├── arc_dataset_meta_test.csv     # Metadati del set di test
│   │   └── split_report.txt              # Report di tracciabilità dello split
│   │
│   ├── estrazione e sottocampionamento/
│   │   └── mat_tools.py                  # ETL da IEEE DataPort (250 kHz) a Kaggle (10 kHz)
│   │
│   ├── build_dataset_new.py              # Generazione sliding window (100 ms) + labeling
│   ├── split_dataset.py                  # GroupShuffleSplit (80/20) per file sorgente
│   └── dataset_leakage_check.py          # Validazione e audit anti-data leakage
│
├── script/
│   ├── causal_current_norm/            # Normalizzazione causale e test ritardo UL 1699B
│   │   ├── causal_norm.py               # Stima causale di I_nom a finestra mobile con gating
│   │   └── detection_delay.py        # Misura del ritardo di rilevamento (K conferme)
│   │
│   ├── voltage_label_creation/         # Analisi di robustezza etichette e innesco da tensione
│   │   ├── voltage_label_robustness.py # Sweep parametri t1_V e baseline solo livello
│   │   ├── voltage_label_evaluation.py# Valutazione modelli per fasce di livello di corrente
│   │   ├── label_sensitivity.py      # Sensibilità parametri ground-truth (bootstrap 95%)
│   │   └── delay_analysis.py           # Diagnostica casi limite con ritardi elevati
│   │
│   ├── energy_analysis/                  # Calcolo energia dissipata dall'arco (Joule)
│   │   └── arc_energy.py               # Integrazione v*i, ritardi reali modelli e attuazione relè
│   │
│   ├── variable_load_analysis/           # Robustezza a carico variabile e decomposizione spettrale
│   │   ├── variable_load.py           # Analisi dell'impatto di variazioni di carico dinamiche
│   │   ├── load_fusion.py            # Valutazione congiunta su scenari di carico eterogenei (file A + file B)
│   │   ├── level_free_detection.py  # Rilevamento basato esclusivamente su pattern AC (waveform/FFT)
│   │   ├── mean_spectrum.py              # Densità spettrale di potenza media per bande temporali
│   │   ├── multichannel_analysis.py        # Framework multicanale (corrente, tensione, potenza)
│   │   └── channel_plot.py             # Visualizzazione e tracciamento delle forme d'onda
│   │
│   ├── test_set_analysis.py              # Valutazione globale test set, metriche e bootstrap
│   ├── make_paper_figures.py             # Generazione automatica grafici e figure riassuntive
│   │
│   ├── training/
│   │   ├── inception_time/
│   │   │   ├── results/
│   │   │   └── train_inceptiontime.py    # Training + export ONNX + calibrazione
│   │   │
│   │   ├── msrcfe/
│   │   │   ├── results/
│   │   │   ├── export_msrcfe/
│   │   │   ├── reduction_results/            # Varianti ridotte (b3, b4, full) per ST Edge AI
│   │   │   ├── train_msrcfe.py           # Training MS-RCFE + Ridge
│   │   │   ├── export_msrcfe.py          # Export ONNX + calibrazione
│   │   │   ├── msrcfe_prune.py           # Ablazione rami e ranking feature
│   │   │   ├── msrcfe_combinations.py            # Analisi 511 combinazioni dei rami (Pareto)
│   │   │   ├── msrcfe_reductions.py       # Generazione ONNX varianti ridotte b3/b4
│   │   │   └── msrcfe_seed.py            # Test stabilita' su seed casuali multipli
│   │   │
│   │   └── multirockethydra/
│   │       ├── train_arc_compare.py      # Training comparativo Ridge/Hydra/ArcNet
│   │       ├── arcnet/
│   │       ├── hydra/
│   │       ├── ridge/
│   │       └── altri file risultati e confronti
│   │
│   └── quantized_models/
│       ├── Inception/
│       │   ├── confronto_modelli.py          # Benchmark FP32 vs INT8 su test set
│       │   └── inceptiontime_PerChannel_quant_calibration_data_npz_1.onnx
│       │
│       └── msrcfe/
│           ├── stm32_float/                  # Report latenze e risorse FP32
│           ├── stm32_int/                    # Report latenze e risorse INT8
│           ├── confronto_modelli.py          # Benchmark FP32 vs INT8 + Ridge su test set
│           ├── conversion_probe.py           # Analisi nodi QDQ e separazione costi I/O
│           ├── resource_measurement.py             # Benchmark latenze e stima Flash/RAM
│           ├── msrcfe_PerChannel_quant_calibration_msrcfe_npz_1.onnx
│           └── msrcfe_unico_PerChannel_quant_calibration_msrcfe_npz_1.onnx
│
├── pipeline.png
├── requirements.txt
└── README.md
```

---

# Installazione

## 1. Clonare il repository

```bash
git clone https://github.com/edge-arc-lab/DC-ARC-Fault.git
cd DC-ARC-Fault
```

## 2. Creare un ambiente virtuale

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

## 3. Installare le dipendenze

```bash
pip install -r requirements.txt
```

---

# Dataset

Il progetto si basa sulla libreria pubblica di riferimento:

**Photovoltaic (PV) DC Arc-Fault Library**  
[IEEE DataPort](https://ieee-dataport.org/open-access/photovoltaic-pv-dc-arc-library)

Il dataset grezzo originale (centinaia di GB a 250 kHz con file accessori spuri) viene ridotto e pulito tramite lo strumento dedicato:
- `dataset/estrazione e sottocampionamento/mat_tools.py`:
  - `clean`: scarta file accessori e replica la gerarchia isolando i soli file `.mat`.
  - `extract`: estrae le variabili fisiche d'interesse (**CurrentData**, **VoltageData**, **power**), sottocampiona da 250 kHz a 10 kHz (fattore di decimazione 25) e salva i file compressi verificandone l'integrità.

Il dataset pulito risultante è disponibile per il download diretto su Kaggle:  
**[Kaggle – DC Arc Fault Dataset](https://www.kaggle.com/datasets/yassirsuarez/dc-arc-fault)**

Scaricare l'archivio da Kaggle ed estrarlo in `dataset/dataset/`. Struttura attesa:

```text
dataset/
└── dataset/
    ├── E001_0050V_00.50A_G001
    ├── E001_0050V_00.50A_G002
    └── ...
```

---

# Pipeline del Progetto

![Pipeline](pipeline.png)

---

# 1. Preprocessing & Physical Split

A partire dai segnali campionati a 10 kHz in `dataset/dataset/`, la generazione del dataset e la suddivisione avvengono tramite due script sequenziali:

### Generazione Finestre (`build_dataset_new.py`)
Applica una strategia a **finestra scorrevole (sliding window)** per emulare l'inferenza real-time embedded:
- **Finestra temporale ($W$):** 100 ms (1.000 campioni a 10 kHz).
- **Passo di scorrimento ($S$):** 50 ms (500 campioni, overlap al 50%).
- **Rilevamento innesco arco ($t_1$):** calcolo della corrente nominale $I_{nom}$ nei primi 200 ms e determinazione di $t_1$ quando la corrente scende sotto il 95% di $I_{nom}$ per almeno 300 ms consecutivi.
- **Labeling supervisionato:**
  - `label = 0` (pre-arco / normale): finestre antecedenti $t_1$.
  - `label = 1` (arco in corso): finestre successive a $t_1$.
- **Bilanciamento:** limite massimo di finestre per classe per file (`MAX_WIN_PER_CLASS`) per evitare sovrarappresentazioni.

Output generati in `dataset/dataset_new/`:
- `arc_dataset_new.npz`: dataset completo (34.931 finestre totali, tensore $X$ e label $y$).
- `arc_dataset_meta_new.csv`: metadati tracciabili per finestra (file di provenienza, intervallo $t_{start}-t_{end}$, $t_1$, corrente nominale, label).
- `arc_dataset_skipped_new.csv`: registro dei file `.mat` scartati con relativa motivazione (es. corrente nominale troppo bassa).

```bash
cd dataset
python build_dataset_new.py dataset/ --out dataset_new/
```

### Split Fisico Anti-Leakage (`split_dataset.py`)
Esegue la partizione 80% Train / 20% Test tramite **`GroupShuffleSplit`** raggruppando per file sorgente (`filename`). In questo modo, **tutte le finestre temporali provenienti dallo stesso esperimento fisico rimangono confinate nello stesso split**, azzerando il rischio di temporal leakage.

Output generati in `dataset/dataset_new/`:
- `arc_dataset_train.npz` (27.770 finestre da 856 file: 10.400 pre-arco, 17.370 arco).
- `arc_dataset_test.npz` (7.161 finestre da 215 file: 2.641 pre-arco, 4.520 arco).
- `arc_dataset_meta_train.csv` e `arc_dataset_meta_test.csv`: tabelle di metadati per split.
- `split_report.txt`: report completo con dettaglio di tutti i file assegnati al test set.

```bash
python split_dataset.py dataset_new/arc_dataset_new.npz dataset_new/arc_dataset_meta_new.csv --test-size 0.20 --out dataset_new/
```

---

# 2. Data Leakage Verification

Lo script `dataset/dataset_leakage_check.py` esegue una suite di audit e validazione indipendente dal modello prima di qualunque training:

1. **Distribuzione classi:** verifica dell'equilibrio tra classi 0 e 1 in train e test.
2. **Controllo duplicati esatti:** verifica che nessun campione $X$ del test set coincida esattamente con campioni del training set.
3. **Distanza euclidea minima:** quantificazione della separazione spaziale minima tra punti di train e test.
4. **Isolamento file sorgente:** verifica formale che l'intersezione tra i file sorgente di train e test sia vuota ($\text{Train} \cap \text{Test} = \emptyset$).
5. **Adversarial Validation:** stima del rischio di memorizzazione tramite un classificatore lineare Ridge.

```bash
python dataset_leakage_check.py dataset_new/arc_dataset_train.npz dataset_new/arc_dataset_test.npz --meta-train dataset_new/arc_dataset_meta_train.csv --meta-test dataset_new/arc_dataset_meta_test.csv
```

---

# 3. Training & Feature Extraction

La cartella `script/training/` contiene le pipeline di training per tre architetture distinte.

## Modelli utilizzati

| Modello | Sigla letteratura | Tipologia | Deploy STM32 |
|---|---|---|---|
| InceptionTime | InceptionTime | Deep Learning end-to-end (CNN temporale addestrata) | Yes: ONNX → ST Edge AI |
| MS-RCFE + Ridge | MS-RCFE | Fixed-kernel ROCKET multi-scala + classificatore lineare | Yes: ONNX → ST Edge AI |
| MultiRocket + varianti | ROCKET / MultiRocket | Feature-based (kernel casuali + ML classico) | Solo sperimentale |

---

## InceptionTime

`train_inceptiontime.py` gestisce in un unico script:

- training con tsai + PyTorch su GPU
- metriche UL1699B (detection rate, false positive rate)
- analisi multi-soglia
- export `inceptiontime.onnx` (shape dinamica)
- generazione dataset di calibrazione per quantizzazione INT8

**Output prodotti:**

```text
risultati_inception/
├── inceptiontime.onnx            ← per ST Edge AI
├── inceptiontime_training.png    ← curve loss/accuracy
├── results_inceptiontime.png     ← confusion matrix, ROC, PR, score dist.
├── inceptiontime_report.txt      ← metriche complete
├── calibration_data.npz          ← calibrazione ST Edge AI (chiave: 'input')
├── calibration_data.npy          ← alternativo
├── calibration_data_flat.npy     ← fallback 2D
├── calibration_labels.npy        ← label per verifica
└── calibration_info.txt          ← istruzioni per ST Edge AI
```

```bash
python train_inceptiontime.py --epochs 50 --out risultati_inception
python train_inceptiontime.py --epochs 50 --n-cal 200 --out risultati_inception
```

---

## MS-RCFE + Ridge

### Cos'è MS-RCFE

**MS-RCFE** è un estrattore di feature ultra-leggero per serie temporali. Utilizza kernel convoluzionali casuali **fissi** (non addestrati) per proiettare il segnale in uno spazio ad alta dimensione, facilitando la classificazione tramite un modello lineare. L'approccio è ispirato al paradigma **ROCKET** (Dempster et al., 2020).

#### Architettura

Il segnale viene elaborato attraverso **9 rami convoluzionali 1D paralleli** basati su diverse scale temporali:

```
Segnale x ∈ ℝ^T
      ↓
9 Conv1D parallele (kernel fissi, non addestrati)
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
Concatenazione  →  z ∈ ℝ^576
      ↓
Ridge Classifier (lineare)  →  ŷ ∈ {0, 1}
```

Per ogni ramo vengono estratte due statistiche globali (**Max** e **Mean**), producendo un vettore finale di **576 feature**. La classificazione viene affidata a un **Ridge Classifier** lineare, che rappresenta l'unica componente ottimizzata durante l'addestramento.

#### Vantaggi principali

- **Addestramento ultra-rapido:** non richiede backpropagation per la parte convoluzionale, in modo analogo a un Extreme Learning Machine (Huang et al., 2006).
- **Analisi multi-scala:** identifica pattern a diverse frequenze e risoluzioni temporali, in linea con l'approccio di MultiRocket (Tan et al., 2022).
- **Design per embedded:** l'uso di pesi fissi riduce drasticamente l'occupazione di memoria, rendendo il modello compatibile con tutta la gamma STM32.

### Training ed export

```bash
python train_msrcfe.py --train train.npz --test test.npz --out results_msrcfe
python export_msrcfe.py msrcfe_bundle.pkl arc_dataset_train.npz --out export_msrcfe
```

**Output prodotti:**

```text
export_msrcfe/
├── msrcfe.onnx               ← feature extractor (kernel fissi) per ST Edge AI
├── ridge_weights.h           ← classificatore Ridge in C
└── calibration_msrcfe.npz   ← dataset calibrazione INT8
```

> **Nota:** viene esportato solo il feature extractor (kernel fissi). Il classificatore Ridge viene implementato in C tramite `ridge_weights.h`. L'inferenza embedded consiste in tre passi: (1) applicazione dei 9 kernel fissi al segnale grezzo, (2) calcolo di max e mean per ogni kernel, (3) prodotto scalare con i pesi Ridge.

#### Studi di Ottimizzazione e Pruning dei Rami

Per massimizzare l'efficienza computazionale su microcontrollori a bassissimo consumo, la cartella include una suite di analisi per la riduzione dei rami convoluzionali:
- **`msrcfe_prune.py`:** ablazione sistematica dei 9 rami e ranking delle feature per identificare configurazioni minime.
- **`msrcfe_combinations.py`:** sweep esaustivo sulle 511 combinazioni possibili per determinare la frontiera di Pareto (accuratezza vs moltiplicazioni MAC).
- **`msrcfe_reductions.py`:** esportazione ONNX e header C per le varianti ridotte a 3 rami (`b3`, -82.4% MAC) e 4 rami (`b4`, -72.5% MAC), salvate in `reduction_results/`.
- **`msrcfe_seed.py`:** test di robustezza statistica variando il seed dei pesi convoluzionali casuali.

---

## MultiRocket + varianti (analisi comparativa)

`train_arc_compare.py` addestra e confronta tre pipeline in un unico run:

| Pipeline | Descrizione |
|---|---|
| MultiRocket + Ridge | Feature extraction + classificatore lineare |
| MultiRocketHydra + Ridge | Kernel Hydra + Ridge interno |
| MultiRocket + PCA + ArcNet | Feature extraction + PCA + rete neurale |

```bash
python train_arc_compare.py train.npz test.npz
python train_arc_compare.py train.npz test.npz --models ridge arcnet
python train_arc_compare.py train.npz test.npz --pca-components 128 --epochs 40
```

**Output prodotti:**

```text
results/
├── ridge/
│   ├── bundle.pkl               modello + transformer + scaler
│   └── config.json              metriche + risorse stimate
├── hydra/
│   ├── bundle.pkl               modello completo
│   └── config.json              metriche
├── arcnet/
│   ├── bundle.pkl               transformer + scaler + pca + model state_dict
│   └── config.json              metriche + risorse stimate
├── comparison.json
├── comparison_metrics.csv
├── comparison_confusion.csv
├── comparison_resources.csv     risorse per componente
├── comparison_report.txt
└── comparison_plots.png         ROC, PR, confusion matrix, metriche
```

> **Nota deploy STM32:** tutte e tre le pipeline usano MultiRocket come preprocessing. MultiRocket non ha un export C/ONNX automatico, quindi nessuna delle tre è deployabile su STM32 senza reimplementare manualmente i kernel in C. Sono state usate esclusivamente per analisi comparativa offline.

---

# 4. Export ONNX & Quantizzazione

Dopo il training, i modelli vengono esportati e quantizzati per il deployment embedded.

| Modello | Export ONNX | Quantizzazione INT8 |
|---|---|---|
| InceptionTime | Yes: automatico nel training | Yes: via ST Edge AI |
| MS-RCFE + Ridge | Yes: `export_msrcfe.py` (solo feature extractor) | Yes: via ST Edge AI |
| MultiRocket + varianti | No: non disponibile | No: non applicabile |

---

# 5. Deployment su ST Edge AI

I modelli ONNX vengono validati tramite ST Edge AI Core / Developer Cloud.

Per InceptionTime e MS-RCFE il flusso è:

1. Importa il modello `.onnx` in ST Edge AI
2. Carica il dataset di calibrazione `.npz` (chiave: `input`)
3. Seleziona quantizzazione INT8 Per-Channel
4. Avvia la quantizzazione → genera `model_int8.onnx`
5. Carica il modello quantizzato nel progetto STM32CubeIDE
6. Esegui inferenza sul test set e confronta con FP32

---

# 6. Normalizzazione Causale & Conformità UL 1699B

Per valutare la robustezza dei modelli in scenari operativi realistici senza conoscenza a priori del segnale, la cartella `script/causal_current_norm/` implementa:

- **Normalizzazione Causale in Tempo Reale (`causal_norm.py`):** sostituisce il valore nominale statico con una stima mobile di $I_{nom}(t)$ calcolata unicamente sui campioni passati (buffer circolare con isteresi e logiche di gating anti-cricchetto), dimostrando che i modelli mantengono elevate prestazioni anche in presenza di variazioni di scala.
- **Misura del Ritardo di Intervento (`detection_delay.py`):** simula l'elaborazione a finestra scorrevole sul test set e misura il ritardo temporale tra l'innesco dell'arco e la decisione positiva confermata su $K$ finestre consecutive ($K=1, 2, 3$), verificando la conformità al vincolo temporale dello standard **UL 1699B** ($t < 2.5\text{ s}$).

---

# 7. Analisi di Robustezza & Ground-Truth da Tensione

Per verificare l'affidabilità delle etichette di ground-truth e analizzare il comportamento dei modelli rispetto alla definizione dell'istante d'innesco dell'arco ($t_1$), la cartella `script/voltage_label_creation/` include:

- **Sensibilità ai Parametri di Etichettatura (`label_sensitivity.py`):** valuta l'impatto delle variazioni dei criteri di soglia e tempo di conferma sulla separazione pre-arco/arco, calcolando intervalli di confidenza bootstrap al 95%.
- **Validazione con Innesco da Tensione (`voltage_label_evaluation.py`):** confronta l'innesco ricavato dalla caduta di tensione del generatore/stringa ($t_{1,V}$) con quello basato sulla corrente, stratificando le performance dei modelli per fasce di intensità di corrente.
- **Robustezza Parametrica (`voltage_label_robustness.py`):** effettua sweep parametrici su soglie di tensione e finestre di conferma, confrontando i modelli con baseline basate su soglie di livello puro ed evidenziando eventuali finestre ambigue.
- **Diagnostica Ritardi (`delay_analysis.py`):** analizza finestra per finestra i casi limite in cui il rilevamento presenta ritardi più marcati, esaminando congiuntamente l'andamento di corrente, tensione e score dei classificatori.

---

# 8. Analisi dell'Energia Dissipata dall'Arco

Per valutare il rischio termico e di innesco incendio secondo lo standard UL 1699B, la cartella `script/energy_analysis/` include:

- **Calcolo dell'Energia Dissipata (`arc_energy.py`):** integra la potenza istantanea $E = \int_{t_1}^{t_1+T_{int}} v_{arco}(t) \cdot i(t) \, dt$ su una griglia di tempi di estinzione sintetici e con compensazione della resistenza serie del circuito.
- **Accoppiamento con i Modelli di Inferenza:** combina i ritardi di rilevazione reali (a soglia di confidenza $K$) dei modelli InceptionTime e MS-RCFE (FP32 e INT8) con il tempo di apertura meccanico del relè (attuazione), quantificando l'energia effettiva prima dell'estinzione.
- **Supporto Innesco da Tensione:** consente di avviare l'integrazione energetica dall'istante transitorio rilevato sulla tensione ($t_{1,V}$), catturando l'energia dissipata anche nei casi di innesco asimmetrico o rallentato.

---

# 9. Analisi a Carico Variabile & Robustezza Spettrale

Nei sistemi fotovoltaici reali, le variazioni di irraggiamento e la commutazione degli inverter introducono fluttuazioni e transitori di corrente che possono confondere i rilevatori basati su semplici soglie di ampiezza. La cartella `script/variable_load_analysis/` valuta sistematicamente la capacità dei modelli di discriminare l'arco in condizioni dinamiche:

- **Impatto delle Variazioni di Carico (`variable_load.py`):** valuta l'accuratezza di rilevazione in presenza di gradini di carico e transitori di commutazione, verificando la separabilità tra dinamiche fisiologiche d'impianto e perturbazioni ad alta frequenza indotte dall'arco.
- **Fusione di Scenari di Carico (`load_fusion.py`):** valuta le prestazioni dei classificatori combinando dataset eterogenei e profili di carico differenti, verificando la generalizzazione cross-load.
- **Rilevamento Indipendente dal Livello DC (`level_free_detection.py`):** isola le componenti AC e le feature morfologiche ad alta frequenza (eliminando l'offset DC), dimostrando che i modelli apprendono la firma intrinseca della scarica d'arco anziché fare affidamento sulla caduta di livello medio della corrente.
- **Analisi e Decomposizione Spettrale (`mean_spectrum.py`):** calcola la densità spettrale di potenza media (PSD) per bande temporali durante le varie fasi (pre-arco, transitorio d'innesco, arco stazionario), quantificando l'energia spettrale nelle bande operative.
- **Framework Multicanale (`multichannel_analysis.py`):** elabora ed estrae feature congiunte da canali multipli (corrente, tensione, potenza istantanea) per massimizzare la tempestività d'intervento e minimizzare i falsi allarmi (FPR).
- **Visualizzazione Forme d'Onda e Canali (`channel_plot.py`):** genera tracciati diagnostici ad alta fedeltà delle forme d'onda temporali e dei relativi segnali sincronizzati.

---

# Risultati Sperimentali

I risultati sperimentali di seguito riportati illustrano le prestazioni di classificazione sul test set, i benchmark hardware misurati su microcontrollori target e le verifiche di conformità normativa:

---

## 1. Risultati di Classificazione sul Test Set

Valutazione su **7.161 finestre di test** (4.520 arco, 2.641 pre-arco) provenienti da **215 esperimenti fisici indipendenti**. Gli intervalli di confidenza al 95% sono ottenuti tramite ricampionamento **bootstrap su 2.000 repliche** a livello di esperimento.

| Modello | Precisione | Detection Rate (DR) [%] | False Positive Rate (FPR) [%] | FN | FP | Accuratezza |
|---|---|---|---|---|---|---|
| **InceptionTime** | FP32 | 99.7% [99.4, 100.0] | 0.61% [0.23, 1.09] | 13 | 16 | **99.60%** |
| **InceptionTime** | INT8 | 98.5% [97.4, 99.3] | 0.15% [0.00, 0.47] | 70 | 4 | **98.97%** |
| **MS-RCFE + Ridge** | FP32 | 99.5% [99.0, 99.9] | 0.11% [0.00, 0.31] | 23 | 3 | **99.64%** |
| **MS-RCFE + Ridge** | INT8 | 99.2% [98.6, 99.6] | 0.11% [0.00, 0.31] | 38 | 3 | **99.43%** |

> **Impatto della quantizzazione INT8:** la quantizzazione post-training riduce l'accuratezza solo di 0.63 punti percentuali per InceptionTime e di appena 0.21 punti per MS-RCFE. Il calo si traduce prevalentemente in mancate rilevazioni (FN) piuttosto che in falsi allarmi (FP), preservando un FPR estremamente basso ($\le 0.15\%$).

---

## 2. Benchmark Hardware STM32 — Latenza di Inferenza Misurata

Latenza di inferenza reale misurata su hardware fisico target per singola finestra da 100 ms (1.000 campioni a 10 kHz):
- **STM32N6570-DK:** microcontrollore con acceleratore hardware **Neural-ART NPU**.
- **STM32H7S78-DK:** microcontrollore general-purpose basato su **ARM Cortex-M7** (600 MHz) senza acceleratore.

| Scheda / Piattaforma | Accelerazione | Modello | FP32 [ms] | INT8 [ms] | Speed-up INT8 | Throughput INT8 |
|---|---|---|---|---|---|---|
| **STM32N6570-DK** | Neural-ART NPU | InceptionTime | 6218.00 ms | **19.73 ms** | **~315×** | **50.68 inf/s** |
| **STM32H7S78-DK** | Cortex-M7 (CPU) | InceptionTime | 6997.67 ms | 2370.64 ms | ~2.9× | 0.42 inf/s |
| **STM32H7S78-DK** | Cortex-M7 (CPU) | MS-RCFE + Ridge | 84.75 ms | **69.77 ms** | ~1.2× | **14.33 inf/s** |

### Risorse di Memoria Embedded

| Parametro / Risorsa | InceptionTime | MS-RCFE + Ridge |
|---|---|---|
| **Ingresso** | Finestra grezza (100 ms, 1.000 campioni) | Finestra grezza (100 ms, 1.000 campioni) |
| **Feature Extractor** | Convoluzioni addestrate end-to-end | $9 \times 32$ kernel casuali fissi (576 feature) |
| **Classificatore** | Softmax | Ridge ($\alpha = 1$, pesi in C) |
| **Operazioni MACs** | — | **1.632 M** |
| **Target Hardware ideale** | STM32N6 (NPU Neural-ART) | STM32H7S78 (Cortex-M7 senza NPU) |
| **Precisione** | FP32, INT8 | FP32, INT8 |
| **RAM Runtime (INT8)** | **2.15 MB** | **≈ 141 KiB** (141.11 KB) |
| **Flash Totale (INT8)** | 399 KB (pesi INT8) | **30.81 KB** (23.56 KB FP32) |

> **Considerazioni chiave sul deployment:**
> 1. **InceptionTime** richiede necessariamente l'NPU: su Cortex-M55 senza NPU impiega 6.22 s, mentre l'acceleratore Neural-ART abbatte la latenza a **19.73 ms** ($\sim 315\times$ di speed-up), rendendolo compatibile con il passo di 50 ms dell'overlap a fronte di 2.15 MB di RAM interna.
> 2. **MS-RCFE + Ridge** richiede solo **141 KiB di RAM** e **30.8 KB di Flash**. Eseguito sul Cortex-M7 senza alcun acceleratore hardware, completa l'inferenza in **69.77 ms** in INT8 (84.75 ms in FP32).

---

## 3. Verifica dei Limiti di Sicurezza UL 1699B

Simulazione a finestra scorrevole condotta su tutti i **442 esperimenti di test contenenti arco**. Il ritardo è misurato dall'istante fisico di innesco $t_1$ al completamento della finestra che conferma $K$ predizioni positive consecutive (escludendo il tempo di attuazione meccanica). L'energia è calcolata integrando $E = \int v(t) i(t) dt$.

### Confronto Indicativo con i Limiti UL 1699B

| Grandezza Misurata | Mediana | Caso Peggiore (Max) | Limite Specificato UL 1699B | Stato |
|---|---|---|---|---|
| **Ritardo di rilevamento ($K = 1$)** | **117 – 143 ms** | **1940 ms** (1.94 s) | — | Conforme |
| **Ritardo con attuazione 50 ms ($K = 3$)** | **0.24 – 0.29 s** | **2.09 s** (2090 ms) | **2.50 s** | **Conforme** |
| **Energia dell'arco ($K = 1$, attuazione 0 ms)** | **9.3 – 13.6 J** | **267.7 J** | **750.0 J** | **Conforme** |

### Energia Dissipata fino alla Rilevazione ($K = 1$, Attuazione 0 ms)

| Modello | Precisione | Mediana [J] | 95° Percentile [J] | Massimo Assoluto [J] |
|---|---|---|---|---|
| **InceptionTime** | FP32 | 9.3 J | 19.1 J | 36.9 J |
| **InceptionTime** | INT8 | 13.6 J | 53.5 J | 267.7 J |
| **MS-RCFE + Ridge** | FP32 | 12.8 J | 30.2 J | 93.5 J |
| **MS-RCFE + Ridge** | INT8 | 12.9 J | 35.2 J | 251.5 J |

---

## 4. Sensibilità ai Parametri di Etichettatura

Valutazione della robustezza dei modelli addestrati (senza riaddestramento) rispetto a variazioni nei criteri di definizione del ground-truth (soglia di ampiezza corrente $I_{nom}$ e finestra di conferma temporale):

| Configurazione Etichette | Inception FP32 (DR / FPR %) | Inception INT8 (DR / FPR %) | MS-RCFE FP32 (DR / FPR %) | MS-RCFE INT8 (DR / FPR %) |
|---|---|---|---|---|
| **Base (Soglia 0.95, Conf. 300 ms)** | **99.7 / 0.6** | **98.5 / 0.2** | **99.5 / 0.1** | **99.2 / 0.1** |
| Soglia 0.90 | 100 / 47.3 | 100 / 40.8 | 100 / 44.0 | 100 / 43.2 |
| Soglia 0.93 | 100 / 21.0 | 100 / 9.2 | 100 / 14.6 | 100 / 13.5 |
| Soglia 0.97 | 96.8 / 0.1 | 93.2 / 0.2 | 95.2 / 0.1 | 94.9 / 0.1 |
| Soglia 0.98 | 96.0 / 0.1 | 92.6 / 0.2 | 94.6 / 0.1 | 94.3 / 0.1 |
| Tempo conferma 100 ms | 99.4 / 0.1 | 97.9 / 0.0 | 99.0 / 0.0 | 98.7 / 0.0 |
| Tempo conferma 1000 ms | 100 / 1.6 | 98.9 / 0.3 | 99.9 / 0.6 | 99.6 / 0.4 |

> Il test evidenzia che i classificatori apprendono con precisione il confine decisionale su cui sono stati supervisionati, confermando l'importanza di analizzare la separabilità fisica delle feature ad alta frequenza rispetto alla sola componente continua.

---

## 5. Ablazione e Pruning dei Rami di MS-RCFE

Studio di riduzione dei 9 rami convoluzionali (valutato su 10 estrattori casuali con seed indipendenti, 9.000 finestre di training) per identificare configurazioni Pareto-ottimali a bassissima latenza:

| Sottoinsieme Rami | DR [%] | Balanced Accuracy [%] | Operazioni MACs | Latenza INT8 (STM32H7S78) | Latenza INT8 (STM32N6570) |
|---|---|---|---|---|---|
| **Tutti i 9 rami (Full)** | **99.32 ± 0.23%** | **99.59 ± 0.10%** | **1.632 M** (100%) | **69.77 ms** | **88.81 ms** |
| 6 rami ($k = 3, 5$) | 98.92 ± 0.66% | 99.41 ± 0.32% | 768 k (47.1%) | — | — |
| **4 rami (`b4`)** | 98.52 ± 1.00% | 99.22 ± 0.49% | 448 k (27.5%) | **26.46 ms** | **32.46 ms** |
| **3 rami (`b3`)** | 98.06 ± 1.05% | 98.99 ± 0.52% | 288 k (17.6%) | **17.78 ms** | **21.13 ms** |
| 1 ramo ($k = 3, d = 2$) | 95.29 ± 2.06% | 97.61 ± 1.03% | 96 k (5.9%) | — | — |
| Ottimo selezionato via CV | 99.30 ± 0.22% | 99.58 ± 0.10% | 22% – 67% | — | — |

> **Impatto pratico:** le varianti ridotte a 3 rami (`b3`: $k=3$, $d=1,2,4$) e 4 rami (`b4`: $k=3$ con $d=1,2,4$ più $k=5$ con $d=2$) abbattono i MACs del 82.4% e 72.5%, riducendo la latenza su Cortex-M7 a **17.78 ms** e **26.46 ms**. Questo consente a MS-RCFE di scendere ampiamente sotto il vincolo temporale dei 50 ms anche su microcontrollori convenzionali senza acceleratori.

---

## 6. Confronto Baseline MultiRocket (Solo Offline)

| Pipeline | Accuracy | Balanced Acc | F1 | ROC-AUC |
|---|---|---|---|---|
| MultiRocket + Ridge | 99.54% | 99.52% | 99.63% | 99.93% |
| MultiRocketHydra + Ridge | **99.65%** | **99.66%** | **99.72%** | 99.66% |
| MultiRocket + PCA + ArcNet | 99.55% | 99.52% | 99.65% | **99.99%** |

> MultiRocketHydra non è stimabile: i kernel Hydra interni non sono accessibili come array.

---

# Conclusioni e Raccomandazioni

Il progetto ha valutato due famiglie principali di modelli per la rilevazione di DC arc fault, con requisiti congiunti di alta accuratezza e compatibilità embedded real-time:

**MS-RCFE + Ridge** è la soluzione ottimale per microcontrollori a basso costo privi di acceleratore (es. STM32H7 con Cortex-M7). Combina un'accuratezza del **99.43%** anche dopo quantizzazione INT8, un'impronta hardware minima (**30.8 KB Flash, 141 KiB RAM**) e un tempo di inferenza di **69.77 ms** in INT8 (84.75 ms in FP32). Con le varianti potate a 3 rami (`b3`) e 4 rami (`b4`), la latenza si riduce ulteriormente a **17.78 ms** e **26.46 ms**, consentendo l'inferenza continua con overlap a 50 ms.

**InceptionTime** è la scelta ad alte prestazioni quando è disponibile un acceleratore hardware dedicato (**Neural-ART NPU su STM32N6570-DK**). L'acceleratore NPU porta la latenza a **19.73 ms** (con un'accelerazione di **~315×** rispetto ai 6218 ms della sola CPU Cortex-M55) e accuratezza del **98.97%**, richiedendo 2.15 MB di RAM interna.

Entrambe le soluzioni garantiscono piena conformità ai requisiti di sicurezza dello standard **UL 1699B**, con ritardi mediani di 117–143 ms ($< 2.5\text{ s}$) ed energia dissipata prima dello sgancio inferiore a 267.7 J ($< 750\text{ J}$).

---

# Autori & Affiliazione

- **Lorenzo Meloccaro** — Università Politecnica delle Marche
- **Yassir Flavio Suarez Sanchez** — Università Politecnica delle Marche
---

# Licenza

Il dataset originale appartiene ai rispettivi autori del progetto pubblicato su IEEE DataPort.
Il codice del repository è distribuito secondo la licenza specificata nel progetto.
