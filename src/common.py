"""
common.py - Fungsi bersama untuk skrip training (baseline dan Keras).

Aturan evaluasi (penting untuk integritas laporan):
- Pemilihan model/hyperparameter HANYA memakai split `val`.
- Laporan akhir: `test` (grouped split, template tidak pernah dilihat saat training)
  dan `manual` (manual_test.csv, ditulis sendiri; sudah dipakai untuk analisis error
  sehingga statusnya data pengembangan, bukan lagi test buta).
- `manual_test_blind.csv` TIDAK disentuh skrip ini. Dibuka sekali di akhir lewat
  evaluate_blind.py setelah model final dipilih.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess  # noqa: E402

LABELS = ["Ready Stock", "Custom Sajadah", "Custom Handuk", "Hampers", "Other"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
DATASET_FILE = "dataset_sintetis_v3.csv"
MANUAL_FILE = "manual_test.csv"


def load_data(data_dir="data", dataset=None):
    """Return (dataframe lengkap, dict split -> dataframe). Split: train, val, test, manual."""
    g = pd.read_csv(os.path.join(data_dir, dataset or DATASET_FILE))
    splits = {s: g[g.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    splits["manual"] = pd.read_csv(os.path.join(data_dir, MANUAL_FILE))
    return g, splits


def prep(texts, **kwargs):
    """Terapkan preprocessing ke daftar teks."""
    return [preprocess(t, **kwargs) for t in texts]


def evaluate(y_true, y_pred):
    """Metrik ringkas: accuracy, macro F1, dan precision/recall/F1 per kelas."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=LABELS)), 4),
        "per_class": {
            l: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                "f1": round(float(f[i]), 3), "support": int(s[i])}
            for i, l in enumerate(LABELS)
        },
    }


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def misclassified(df, y_pred, n=None):
    """DataFrame baris yang salah prediksi (untuk analisis error)."""
    out = df.assign(pred=y_pred)
    out = out[out.label != out.pred][["text", "label", "pred"]]
    return out if n is None else out.head(n)
