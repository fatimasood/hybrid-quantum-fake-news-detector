# Beyond Classical NLP: Hybrid Quantum Fake News Detector (v1.0)

[![Framework - PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Quantum - PennyLane](https://img.shields.io/badge/Quantum-PennyLane-6100a3?style=flat-square)](https://pennylane.ai/)
[![Transformer - HuggingFace](https://img.shields.io/badge/Transformer-HuggingFace-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/)

An advanced **Hybrid Quantum-Classical Machine Learning** framework designed for binary fake news classification. This repository implements a cutting-edge pipeline fusing contextual classical text embeddings from **DistilBERT** with a **Variational Quantum Circuit (VQC)** processed via an explicit feature attention fusion head.

Tested and validated on the highly nuanced **LIAR Dataset**.

---

## 🧬 Architecture Overview

The model leverages a dual-topology pipeline that bridges State-of-the-Art (SOTA) classical transformers with variational quantum computing:

1. **Classical Text Encoding:** Sentences are tokenized and passed through `distilbert-base-uncased` to extract rich contextual hidden states ($768$-dimensional embedding via the `[CLS]` token).
2. **Quantum Feature Processing:** The classical embedding is projected down via a linear layer and fed into a **6-Qubit Variational Quantum Circuit** using Pauli-Y (`RY`) embedding. The circuit runs 2 layers of variational rotations (`RX`, `RY`, `RZ`) interleaved with a linear CNOT entanglement chain.
3. **Attention-Based Feature Fusion:** The classical $768$-dimensional vectors and the $64$-dimensional quantum circuit outputs are concatenated and optimized utilizing a customized self-attention pooling mechanism.
4. **Classification Head:** A deep neural network optimized with **Weighted BCE Loss** outputs the final binary decision (`Real` vs `Fake`).

---

## 📊 Experimental Results (LIAR Dataset)

The model was evaluated on the binary configuration of the LIAR dataset (fusing pants-fire, false, barely-true into `Fake` and half-true, mostly-true, true into `Real`).

### Core Metrics
* **Test Accuracy:** `63.20%`
* **Test F1-Score (Binary):** `68.20%`
* **AUC-ROC Score:** `0.6225`
* **Matthews Correlation Coefficient (MCC):** `0.2453`

### Binary Classification Report
```text
              precision    recall  f1-score   support

    Real (0)     0.5665    0.5601    0.5633       973
    Fake (1)     0.6792    0.6848    0.6820      1323

    accuracy                         0.6320      2296
   macro avg     0.6228    0.6225    0.6226      2296
weighted avg     0.6314    0.6320    0.6317      2296

🛠️ Tech Stack & Dependencies
Deep Learning Framework: PyTorch (v2.3.0)
Quantum Simulator: PennyLane (v0.36.0) & PennyLane-Lightning
Transformers: Hugging Face transformers (v4.41.2)
Classical Metrics: Scikit-Learn, Pandas, NumPy
