"""
predictor.py - Pembungkus inferensi untuk model final (ensemble CNN + LogReg).

Dipakai oleh FastAPI (endpoint /classify) dan evaluate_blind.py.

    from predictor import OrderClassifier
    clf = OrderClassifier("models/final")
    clf.predict("Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah")
    # {'label': 'Custom Sajadah', 'confidence': 0.93, 'needs_review': False, 'probabilities': {...}}

`needs_review` = True bila confidence < review_threshold (di ensemble_config.json).
Nilai awalnya 0.6 dan BELUM dikalibrasi; sesuaikan setelah melihat data nyata.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import tensorflow as tf

from preprocessing import preprocess


class OrderClassifier:
    def __init__(self, model_dir="models/final"):
        with open(os.path.join(model_dir, "ensemble_config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.labels = self.config["labels"]
        self.w_cnn = float(self.config["cnn_weight"])
        self.threshold = float(self.config.get("review_threshold", 0.6))
        self.lr = joblib.load(os.path.join(model_dir, self.config["lr_file"]))
        self.cnns = [tf.keras.models.load_model(os.path.join(model_dir, f), compile=False)
                     for f in self.config["cnn_files"]]
        # urutan kolom predict_proba LogReg (alfabet) -> urutan self.labels
        self._lr_order = [list(self.lr.classes_).index(l) for l in self.labels]

    def predict_proba(self, texts):
        """Return (ensemble, cnn, logreg): tiga array (n, n_label) dengan urutan kolom = self.labels."""
        x = [preprocess(t) for t in texts]
        p_lr = self.lr.predict_proba(np.array(x))[:, self._lr_order]
        p_cnn = np.mean([m.predict(tf.constant(x), verbose=0, batch_size=128) for m in self.cnns], axis=0)
        return self.w_cnn * p_cnn + (1 - self.w_cnn) * p_lr, p_cnn, p_lr

    def predict_many(self, texts):
        probs, _, _ = self.predict_proba(list(texts))
        out = []
        for p in probs:
            i = int(p.argmax())
            out.append({
                "label": self.labels[i],
                "confidence": round(float(p[i]), 4),
                "needs_review": bool(p[i] < self.threshold),
                "probabilities": {l: round(float(v), 4) for l, v in zip(self.labels, p)},
            })
        return out

    def predict(self, text):
        return self.predict_many([text])[0]
