## Team Members

- Shuo Ma– Model A (BiLSTM Cross-Attention)
- Kunhong Zou – Model B (ESIM-style BiGRU)
- Mohaimen Rashid – Model C (Lightweight Transformer)

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

## Implementation

All models are implemented in **PyTorch**, trained **from scratch**, and run on **Google Colab**.
Evaluation includes accuracy comparison, attention ablation, and qualitative visualization.

---
