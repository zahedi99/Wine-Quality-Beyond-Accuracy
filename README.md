# Improving Reliability & Interpretability in Predictive AI (Wine Quality)

A **trustworthy tabular ML study** comparing **XGBoost** (gradient-boosted trees) vs **TabNet**
(attention-based deep tabular network) for **red wine quality** prediction — with emphasis on:

- **Predictive performance** (accuracy, macro-F1, balanced accuracy)
- **Interpretability** (SHAP + PDP/ICE)
- **Probability reliability** (**calibration**, reliability diagrams, **ECE**)

> MSc Artificial Intelligence & Robotics (University of Hertfordshire) — **Distinction project**

---

## Quick Results (3-class formulation)

The original 6-class labels (3–8) are highly imbalanced and poorly separable.
The project reformulates the task into:

- **Low**: 3–4
- **Medium**: 5–6
- **High**: 7–8

**Final headline comparison (held-out test set):**

| Model | Accuracy | Macro-F1 | Balanced Acc | ECE (↓) |
|------|----------:|---------:|-------------:|--------:|
| XGBoost | **0.8812** | **0.6685** | **0.6433** | **0.0339** |
| TabNet  | 0.8719 | 0.5572 | 0.5194 | 0.0631 |

Key takeaway: On this small, moderately imbalanced tabular dataset, **XGBoost** achieves the most
dependable balance of **accuracy + interpretability + calibration**.

---

## Repo Contents

- `notebooks/` – original project notebooks (renamed for clarity)
- `src/` – small, importable utilities (ECE + preprocessing helpers)
- `reports/` – MSc final project report PDF (full methodology + discussion)

---

## How to Run (no heavy experiments required)

Create an environment:

### Option A — Conda
```bash
conda env create -f environment.yml
conda activate wine-quality-reliability
```

### Option B — pip
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Then open the notebooks:
```bash
jupyter notebook
```

---

## What Makes This Project Different

Most “wine quality” repos stop at accuracy. This one explicitly evaluates **trustworthiness**:

- Interprets model reasoning using **SHAP** (global + class-wise)
- Validates feature effects using **PDP** and **ICE**
- Measures probability reliability using **reliability diagrams** + **Expected Calibration Error**
- Tests imbalance strategies (**SMOTE-ENN**, undersampling variants) and explains why **GAN augmentation** failed

---

## Dataset

UCI / Cortez et al. Red Wine Quality dataset (1599 rows, 11 physicochemical features).

---

## Citation

If you reference this work, please cite the included report:
- `reports/FPR.pdf`

---

## Author

**Parsa Zahedi**
- GitHub: @zahedi99 (add link in your profile)
- LinkedIn: linkedin.com/in/parsa-zahedi
