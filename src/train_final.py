"""
train_final.py - Latih dan BEKUKAN model final: ensemble CNN (Keras, dari nol) + TF-IDF LogReg.

Konfigurasi sama persis dengan yang dievaluasi di GroupKFold (compare_models.py --cv):
  - CNN 1D tanpa augmentasi, 10 epoch, rata-rata probabilitas 3 seed
  - TF-IDF (1,2)-gram + Logistic Regression (C=1)
  - Ensemble: 0.5 * CNN + 0.5 * LogReg
Dilatih di SELURUH data generator (train + val + test) karena evaluasi sudah dilakukan lewat
GroupKFold dan test set manual/blind. manual_test.csv dan manual_test_blind.csv tidak dipakai
untuk training.

Jalankan dari root proyek:
    python src/train_final.py
Hasil: models/final/ (3 file .keras, baseline_lr.joblib, ensemble_config.json)
"""
import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from common import DATASET_FILE, LABEL2ID, LABELS, evaluate, load_data, prep
from predictor import OrderClassifier
from train_baseline import make_pipeline
from train_keras import build_model


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model-dir", default="models/final")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--cnn-weight", type=float, default=0.5)
    args = ap.parse_args()

    t0 = time.time()
    full, sp = load_data(args.data_dir, args.dataset)
    texts = prep(full.text)
    y_int = full.label.map(LABEL2ID).values
    os.makedirs(args.model_dir, exist_ok=True)

    # --- LogReg ---
    lr = make_pipeline(LogisticRegression(C=1, max_iter=5000, random_state=42)).fit(np.array(texts), full.label.values)
    joblib.dump(lr, os.path.join(args.model_dir, "baseline_lr.joblib"))
    print("LogReg selesai")

    # --- CNN x seed ---
    cnn_files = []
    for seed in range(args.seeds):
        tf.keras.utils.set_random_seed(42 + seed)
        m = build_model("cnn", texts)
        m.fit(tf.constant(texts), y_int, epochs=args.epochs, batch_size=32, verbose=0)
        name = f"cnn_seed{42 + seed}.keras"
        m.save(os.path.join(args.model_dir, name))
        cnn_files.append(name)
        print(f"CNN seed {42 + seed} selesai")

    ds_path = os.path.join(args.data_dir, args.dataset or DATASET_FILE)
    config = {
        "version": "final-1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": LABELS,
        "dataset": {"file": os.path.basename(ds_path), "rows": int(len(full)), "sha256_16": sha256(ds_path)},
        "preprocessing": "src/preprocessing.py preprocess() default (negasi, fuzzy typo, masking)",
        "lr_file": "baseline_lr.joblib",
        "lr": {"C": 1, "tfidf": "word 1-2gram, sublinear_tf, min_df=2"},
        "cnn_files": cnn_files,
        "cnn": {"epochs": args.epochs, "seeds": args.seeds, "augmentation": False, "batch_size": 32},
        "cnn_weight": args.cnn_weight,
        "review_threshold": 0.6,
        "review_threshold_note": "nilai awal, belum dikalibrasi; sesuaikan setelah melihat data nyata",
        "trained_on": "seluruh data generator (train+val+test); manual & blind tidak dipakai training",
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "scikit-learn": sklearn.__version__, "numpy": np.__version__},
    }
    with open(os.path.join(args.model_dir, "ensemble_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # --- sanity check: muat ulang dari disk dan uji pada manual_test.csv (data pengembangan) ---
    clf = OrderClassifier(args.model_dir)
    man = sp["manual"]
    pred = clf.predict_many(list(man.text))
    res = evaluate(man.label.values, np.array([p["label"] for p in pred]))
    print(f"\nSanity check (manual_test, data pengembangan): macro F1 = {res['macro_f1']:.3f}, acc = {res['accuracy']:.3f}")
    print(f"Model beku tersimpan di {args.model_dir}/ ({time.time() - t0:.0f} detik)")


if __name__ == "__main__":
    main()
