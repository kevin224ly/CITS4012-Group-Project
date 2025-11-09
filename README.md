## Team Members

- Shuo Ma– Model A (BiLSTM Cross-Attention)
- Kunhong Zou – Model B (ESIM-style BiGRU)
- Mohaimen Rashid – Model C (Lightweight Transformer)

---

## Project Overview

This repository contains the full, runnable code used for the CITS4012 Group 35 submission. The primary entry point is the notebook `CITS4012_35.ipynb`, which trains and evaluates three models on the provided science-domain NLI dataset and produces figures (including attention visualisations) reported in the paper.

If a marker needs to run the code, follow the Quickstart below. The notebook can run locally or in Google Colab. No private credentials are required.

---

## Models Overview

### **Model A – BiLSTM Cross-Attention**

- Static word embeddings (GloVe or word2vec)
- BiLSTM encoders for premise and hypothesis
- Bilinear cross-attention and pooled interaction vector
- MLP classifier
  → Serves as interpretable baseline with attention visualization.

### **Model B – ESIM-Style BiGRU with Inference Composition**

- Shared BiGRU encoders
- Soft alignment attention
- Local inference enhancement (difference/product)
- Second BiGRU for inference composition
- Pooling (average and max) → classifier
  → Highlights alignment reasoning and allows rich ablation points.

### **Model C – Lightweight Transformer Cross-Encoder**

- Learned token + segment embeddings
- 2–4 layer Transformer encoder (multi-head self-attention)
- [CLS] representation → classifier
  → Demonstrates transformer architecture under limited compute.

---

## Repository Layout

- `CITS4012_35.ipynb` – end-to-end pipeline (data checks, preprocessing, training, evaluation, plots)
- `data/` – expected location of `train.json`, `validation.json`, `test.json`
- `results/` – metrics, checkpoints and artefacts are written here by the notebook
- `diagrams/` – generated architecture diagrams and attention heatmaps (also used in the paper)
- `CITS4012_35_architecture_diagrams.py` – script to render the three architecture diagrams (Graphviz)
- `requirements.txt` – pinned runtime dependencies

---

## Quickstart (Local execution)

1) Python

- Use Python 3.10–3.11 (tested on macOS 26). Create a clean virtual environment.

```bash
cd "CITS4012-Group-Project"
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Dataset

- Place the three JSON files under `data/` with exact names:
  - `data/train.json`
  - `data/validation.json`
  - `data/test.json`
- If the files are missing, the notebook contains an optional Colab-only downloader; for local runs just copy the provided splits into the folder above.

3) Launch Jupyter and run the notebook

```bash
jupyter lab
# then open CITS4012_35.ipynb and Run All
```

The notebook will:

- normalise and load the dataset
- train Models A/B/C from scratch
- save metrics/artefacts to `results/`
- render attention/diagnostic plots used in the paper

Optional non-interactive execution (runs all cells and writes a new executed notebook):

```bash
jupyter nbconvert --to notebook --execute CITS4012_35.ipynb --output CITS4012_35_executed.ipynb
```

---

## Quickstart (Google Colab)

1) Upload the whole `CITS4012-Group-Project` folder to your Google Drive or open `CITS4012_35.ipynb` directly in Colab.
2) Run the notebook top-to-bottom. The first setup cell installs required packages in Colab automatically; local environments skip that step.
3) If dataset JSON files are missing, the Colab downloader cell can fetch them (or you may upload them to the `data/` folder).

---

## How to reproduce key artefacts required by the paper

- Train/evaluate all models: run `CITS4012_35.ipynb` end-to-end.
- Model A attention heatmap (Figure – bilinear cross-attention): run the “Model A … attention visualisation” cell block; the plot is also saved to `diagrams/`.
- Architecture diagrams: ensure Graphviz is installed locally and run:

```bash
# macOS (Graphviz binary)
brew install graphviz  # if not installed

# Python package used by the diagram script
pip install graphviz

# Generate PNGs into diagrams/
python CITS4012_35_architecture_diagrams.py
```

The resulting files are written as:

- `diagrams/modelA_architecture.png`
- `diagrams/modelB_architecture.png`
- `diagrams/modelC_architecture.png`

---

## Minimal run instructions for markers (as requested)

If it’s not apparent from the notebook itself, the minimal steps to run our program are:

```bash
cd "CITS4012-Group-Project"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Ensure data files exist at data/{train,validation,test}.json
jupyter nbconvert --to notebook --execute CITS4012_35.ipynb --output CITS4012_35_executed.ipynb
```

Outputs you can expect under `results/` include per-model metrics and (when enabled) checkpoints; plots (attention, confusion matrices, etc.) are produced inline and saved under `diagrams/`.

---

## Troubleshooting

- Graphviz errors when generating diagrams: install the system binary (`brew install graphviz`) and the Python package (`pip install graphviz`).
- TensorFlow or PyTorch version conflicts: use the provided `requirements.txt` inside a fresh virtual environment.
- Long runtimes on CPU: use Colab (GPU) to match the timings we report.

---

## License

See `LICENSE` for details.
