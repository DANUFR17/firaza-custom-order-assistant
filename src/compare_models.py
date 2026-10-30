"""
compare_models.py - Bandingkan kandidat model final di dataset yang sama, dengan protokol yang sama.

Kandidat:
  1. TF-IDF + Logistic Regression (baseline)
  2. CNN 1D (rata-rata probabilitas 3 seed)
  3. CNN 1D + augmentasi token (rata-rata probabilitas 3 seed)
  4. Ensemble: bobot w * CNN(+aug) + (1 - w) * LogReg, w dipilih dari val

Aturan: pemilihan (varian CNN dan bobot ensemble) HANYA memakai split `val`.
`test` dan `manual` hanya dilaporkan. `manual_test_blind.csv` tidak disentuh.

Jalankan dari root proyek:
    python src/compare_models.py          # perbandingan di split val/test/manual
    python src/compare_models.py --cv     # GroupKFold 5 lipatan per template (lebih stabil)
"""
import argparse
import os
import random
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tensorflow.keras import callbacks

from common import LABEL2ID, LABELS, evaluate, load_data, plot_confusion, prep, save_json
from train_baseline import make_pipeline
from train_keras import augment_text, build_model


def keras_probs(texts_by_split, X, y, train_x, train_y, seeds, epochs=40):
    """Latih CNN per seed; kembalikan rata-rata probabilitas tiap split."""
    acc = {k: [] for k in texts_by_split}
    for seed in range(seeds):
        tf.keras.utils.set_random_seed(42 + seed)
        m = build_model("cnn", X["train"])
        m.fit(tf.constant(train_x), train_y, epochs=epochs, batch_size=32, verbose=0,
              validation_data=(tf.constant(X["val"]), y["val"]),
              callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
        for k in texts_by_split:
            acc[k].append(m.predict(tf.constant(X[k]), verbose=0, batch_size=128))
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def cv_compare(full, seeds=2, epochs=10, folds=5):
    """GroupKFold per template pada seluruh data generator. Lebih stabil daripada satu split val/test
    yang hanya berisi beberapa template. CNN dilatih dengan jumlah epoch tetap (tanpa val)."""
    from sklearn.model_selection import GroupKFold
    X = np.array(prep(full.text), dtype=object)
    y = full.label.map(LABEL2ID).values
    groups = full.template_id.values
    rows = []
    for fi, (a, b) in enumerate(GroupKFold(folds).split(X, y, groups), 1):
        xa, xb = list(X[a]), list(X[b])
        lr = make_pipeline(LogisticRegression(C=1, max_iter=5000, random_state=42)).fit(np.array(xa), y[a])
        p_lr = lr.predict_proba(np.array(xb))
        rng = random.Random(fi)
        xa_aug, ya_aug = xa + [augment_text(t, rng) for t in xa], np.concatenate([y[a], y[a]])
        pc, pa = [], []
        for seed in range(seeds):
            for store, tx, ty in ((pc, xa, y[a]), (pa, xa_aug, ya_aug)):
                tf.keras.utils.set_random_seed(100 + seed)
                m = build_model("cnn", xa)
                m.fit(tf.constant(tx), ty, epochs=epochs, batch_size=32, verbose=0)
                store.append(m.predict(tf.constant(xb), verbose=0, batch_size=128))
        pc, pa = np.mean(pc, axis=0), np.mean(pa, axis=0)
        cand = {"TF-IDF + LogReg": p_lr, "CNN": pc, "CNN + augmentasi": pa,
                "Ensemble CNN+LogReg (0.5/0.5)": 0.5 * pc + 0.5 * p_lr,
                "Ensemble CNN+aug+LogReg (0.5/0.5)": 0.5 * pa + 0.5 * p_lr}
        for name, p in cand.items():
            rows.append({"fold": fi, "model": name, "macro_f1": f1_score(y[b], p.argmax(1), average="macro")})
        print(f"fold {fi}/{folds} selesai", flush=True)
    df = pd.DataFrame(rows)
    return df.groupby("model", sort=False).macro_f1.agg(["mean", "std"]).round(3)


def to_labels(probs):
    return np.array([LABELS[i] for i in probs.argmax(1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cv", action="store_true", help="GroupKFold per template, bukan split val/test")
    args = ap.parse_args()

    t0 = time.time()
    full, sp = load_data(args.data_dir, args.dataset)
    if args.cv:
        res = cv_compare(full)
        os.makedirs(args.out_dir, exist_ok=True)
        res.to_csv(os.path.join(args.out_dir, "model_comparison_groupkfold.csv"))
        print("\nGroupKFold(5) macro F1 per template (mean, std antar lipatan):\n")
        print(res.to_string())
        print(f"\nSelesai dalam {time.time() - t0:.0f} detik.")
        return
    splits = ("val", "test", "manual")
    X = {k: prep(v.text) for k, v in sp.items()}
    y = {k: v.label.map(LABEL2ID).values for k, v in sp.items()}
    ytxt = {k: v.label.values for k, v in sp.items()}

    # --- 1. baseline LogReg (probabilitas urut sesuai LABELS) ---
    lr = make_pipeline(LogisticRegression(C=1, max_iter=5000, random_state=42)).fit(np.array(X["train"]), ytxt["train"])
    order = [list(lr.classes_).index(l) for l in LABELS]
    p_lr = {k: lr.predict_proba(np.array(X[k]))[:, order] for k in splits}

    # --- 2 & 3. CNN tanpa/dengan augmentasi ---
    p_cnn = keras_probs(splits, X, y, X["train"], y["train"], args.seeds)
    rng = random.Random(0)
    aug_x = X["train"] + [augment_text(t, rng) for t in X["train"]]
    aug_y = np.concatenate([y["train"], y["train"]])
    p_aug = keras_probs(splits, X, y, aug_x, aug_y, args.seeds)

    cands = {"TF-IDF + LogReg": p_lr, "CNN (ens. seed)": p_cnn, "CNN + augmentasi (ens. seed)": p_aug}

    # --- 4. ensemble: pilih varian CNN dan bobot dari val ---
    best = None
    for cname, pc in (("CNN", p_cnn), ("CNN+aug", p_aug)):
        for w in (0.25, 0.5, 0.75):
            f = f1_score(ytxt["val"], to_labels(w * pc["val"] + (1 - w) * p_lr["val"]), average="macro")
            if best is None or f > best[0]:
                best = (f, cname, w, pc)
    _, ens_name, ens_w, ens_pc = best
    cands[f"Ensemble ({ens_name} {ens_w:.2f} + LogReg {1 - ens_w:.2f})"] = {
        k: ens_w * ens_pc[k] + (1 - ens_w) * p_lr[k] for k in splits}

    rows, metrics = [], {}
    for name, probs in cands.items():
        r = {"model": name}
        metrics[name] = {}
        for k in splits:
            res = evaluate(ytxt[k], to_labels(probs[k]))
            metrics[name][k] = res
            r[k] = res["macro_f1"]
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "model_comparison.csv"), index=False)
    save_json({"ensemble_choice": {"cnn_variant": ens_name, "weight_cnn": ens_w, "selected_on": "val macro F1"},
               "metrics": metrics}, os.path.join(args.out_dir, "model_comparison.json"))
    print("\nMacro F1 (probabilitas dirata-rata antar seed untuk CNN):\n")
    print(df.to_string(index=False))
    print(f"\nEnsemble dipilih dari val: {ens_name}, bobot CNN = {ens_w}")
    print(f"Selesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
