"""
train_baseline.py - Baseline klasifikasi order Firaza: TF-IDF + model linear (scikit-learn).

Jalankan dari root proyek:
    python src/train_baseline.py --data-dir data --out-dir reports --model-dir models
    python src/train_baseline.py --ablation        # tambahan: ablation langkah preprocessing

Yang dilakukan:
 1. Preprocessing (src/preprocessing.py) lalu TF-IDF unigram+bigram.
 2. Grid kecil (Logistic Regression dan LinearSVC), dipilih berdasarkan macro F1 di `val`.
 3. Evaluasi di `test` (grouped) dan `manual`; confusion matrix; GroupKFold 5 lipatan.
 4. Simpan model (joblib) + metrik (JSON) + daftar error.
Model dilatih dari nol dari data sintetis; tidak memakai model pra-latih.
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import LABELS, evaluate, load_data, misclassified, plot_confusion, prep, save_json
from preprocessing import preprocess


def make_pipeline(clf):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b\w+\b")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def candidates():
    for c in (1, 3, 10, 30):
        yield f"LogReg(C={c})", lambda c=c: LogisticRegression(C=c, max_iter=5000, random_state=42)
    for c in (0.3, 1, 3):
        yield f"LinearSVC(C={c})", lambda c=c: LinearSVC(C=c, random_state=42)


def group_kfold(texts, y, groups, make_clf, k=5):
    scores = []
    for a, b in GroupKFold(k).split(texts, y, groups):
        pipe = make_pipeline(make_clf()).fit(texts[a], y[a])
        scores.append(f1_score(y[b], pipe.predict(texts[b]), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dataset", default=None, help="nama file dataset (default: dataset_sintetis_v3.csv)")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    full, sp = load_data(args.data_dir, args.dataset)
    X = {k: np.array(prep(v.text)) for k, v in sp.items()}
    y = {k: v.label.values for k, v in sp.items()}

    # --- pilih model berdasarkan val ---
    results = []
    for name, make in candidates():
        pipe = make_pipeline(make()).fit(X["train"], y["train"])
        val_f1 = f1_score(y["val"], pipe.predict(X["val"]), average="macro")
        results.append((val_f1, name, make))
        print(f"{name:20s} val macro F1 = {val_f1:.3f}")
    val_f1, best_name, best_make = max(results, key=lambda r: r[0])
    print(f"\nTerpilih (berdasarkan val): {best_name}  (val macro F1 = {val_f1:.3f})")

    pipe = make_pipeline(best_make()).fit(X["train"], y["train"])
    metrics = {"model": best_name, "selection": "val macro F1", "val_macro_f1": round(val_f1, 4)}
    for split in ("val", "test", "manual"):
        pred = pipe.predict(X[split])
        metrics[split] = evaluate(y[split], pred)
        plot_confusion(y[split], pred, f"Baseline {best_name} - {split}",
                       os.path.join(args.out_dir, f"baseline_cm_{split}.png"))
        misclassified(sp[split], pred).to_csv(os.path.join(args.out_dir, f"baseline_errors_{split}.csv"), index=False)
        print(f"{split:7s} macro F1 = {metrics[split]['macro_f1']:.3f}  acc = {metrics[split]['accuracy']:.3f}")

    # --- GroupKFold per template (seluruh dataset generator) ---
    Xall = np.array(prep(full.text))
    gm, gs = group_kfold(Xall, full.label.values, full.template_id.values, best_make)
    metrics["group_kfold_5"] = {"mean": round(gm, 4), "std": round(gs, 4)}
    print(f"GroupKFold(5) macro F1 = {gm:.3f} +- {gs:.3f}")

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.model_dir, "baseline_tfidf.joblib"))
    save_json(metrics, os.path.join(args.out_dir, "baseline_metrics.json"))

    # --- ablation preprocessing (opsional) ---
    if args.ablation:
        variants = {
            "A mentah (tanpa preprocessing)": lambda t: t,
            "B + slang + masking": lambda t: preprocess(t, negation=False, fuzzy=False),
            "C + koreksi typo": lambda t: preprocess(t, negation=False, fuzzy=True),
            "D + negasi (pipeline penuh)": lambda t: preprocess(t, negation=True, fuzzy=True),
        }
        rows = []
        for name, f in variants.items():
            Xv = {k: np.array([f(t) for t in v.text]) for k, v in sp.items()}
            p = make_pipeline(best_make()).fit(Xv["train"], y["train"])
            Xa = np.array([f(t) for t in full.text])
            m, s = group_kfold(Xa, full.label.values, full.template_id.values, best_make)
            rows.append({"varian": name,
                         "val": round(f1_score(y["val"], p.predict(Xv["val"]), average="macro"), 3),
                         "test": round(f1_score(y["test"], p.predict(Xv["test"]), average="macro"), 3),
                         "manual": round(f1_score(y["manual"], p.predict(Xv["manual"]), average="macro"), 3),
                         "groupkfold_mean": round(m, 3), "groupkfold_std": round(s, 3)})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.out_dir, "ablation_preprocessing.csv"), index=False)
        print("\nAblation preprocessing (macro F1):\n", df.to_string(index=False))

    print(f"\nSelesai dalam {time.time() - t0:.0f} detik. Model: {args.model_dir}/baseline_tfidf.joblib")


if __name__ == "__main__":
    main()
