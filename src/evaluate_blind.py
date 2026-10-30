"""
evaluate_blind.py - Evaluasi model beku pada manual_test_blind.csv. HANYA DIJALANKAN SEKALI.

Aturan:
  - Model sudah dibekukan (models/final, train_final.py) SEBELUM file ini dijalankan.
  - Angka resmi di laporan = hasil ensemble beku. Skor komponen (CNN saja, LogReg saja) ditampilkan
    hanya sebagai informasi untuk analisis; JANGAN mengganti model berdasarkan angka blind.
  - Setelah dijalankan, dibuat reports/blind_evaluation.lock.json. Menjalankan lagi akan ditolak
    kecuali memakai --force (dan itu harus dicatat di laporan: blind set tidak lagi buta).
  - --dry-run untuk menguji kode memakai file lain; tidak menulis lock dan menyimpan keluaran
    ke reports/dry_run/.

Jalankan dari root proyek:
    python src/evaluate_blind.py
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import LABELS, evaluate, misclassified, plot_confusion, save_json
from predictor import OrderClassifier


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--blind-file", default="manual_test_blind.csv")
    ap.add_argument("--model-dir", default="models/final")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_dir, "dry_run") if args.dry_run else args.out_dir
    lock = os.path.join(args.out_dir, "blind_evaluation.lock.json")
    if os.path.exists(lock) and not (args.force or args.dry_run):
        with open(lock, encoding="utf-8") as f:
            info = json.load(f)
        print("DITOLAK: test set blind sudah pernah dievaluasi "
              f"({info['evaluated_at']}, macro F1 = {info['macro_f1']}).")
        print("Mengulang berarti blind set tidak lagi buta. Pakai --force hanya jika benar-benar perlu, "
              "dan catat di laporan.")
        sys.exit(1)

    path = os.path.join(args.data_dir, args.blind_file)
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        sys.exit(f"Kolom tidak ada di {path}: {missing}")
    bad = set(df.label) - set(LABELS)
    if bad:
        sys.exit(f"Label tidak valid di {path}: {bad}")

    clf = OrderClassifier(args.model_dir)
    p_ens, p_cnn, p_lr = clf.predict_proba(list(df.text))
    to_lab = lambda p: np.array([LABELS[i] for i in p.argmax(1)])
    pred = to_lab(p_ens)
    y = df.label.values

    res = evaluate(y, pred)
    info_only = {"cnn_saja": evaluate(y, to_lab(p_cnn))["macro_f1"], "logreg_saja": evaluate(y, to_lab(p_lr))["macro_f1"]}

    os.makedirs(out_dir, exist_ok=True)
    plot_confusion(y, pred, "Model final (ensemble) - test blind", os.path.join(out_dir, "blind_cm.png"))
    err = misclassified(df, pred)
    conf = p_ens.max(1)
    err = err.assign(confidence=np.round(conf[(y != pred)], 3))
    err.to_csv(os.path.join(out_dir, "blind_errors.csv"), index=False)
    save_json({
        "model_final": clf.config["version"], "model_created_at": clf.config["created_at"],
        "blind_file": args.blind_file, "blind_sha256_16": file_sha(path), "n": int(len(df)),
        "ensemble": res, "informasi_saja_bukan_untuk_memilih_model": info_only,
        "needs_review_threshold": clf.threshold,
        "needs_review_rate": round(float((conf < clf.threshold).mean()), 3),
    }, os.path.join(out_dir, "blind_metrics.json"))

    print(f"\nTest blind: n = {len(df)}")
    print(f"Ensemble (model final)  macro F1 = {res['macro_f1']:.3f}   accuracy = {res['accuracy']:.3f}")
    print(f"  [info] CNN saja = {info_only['cnn_saja']:.3f}, LogReg saja = {info_only['logreg_saja']:.3f}  (jangan dipakai memilih model)")
    print("\nPer kelas:")
    for l, m in res["per_class"].items():
        print(f"  {l:15s} P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  (n={m['support']})")
    print(f"\n{len(err)} salah dari {len(df)}. Detail: {out_dir}/blind_errors.csv")
    if len(err):
        print(err.to_string(index=False, max_colwidth=90))

    if not args.dry_run:
        with open(lock, "w", encoding="utf-8") as f:
            json.dump({"evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "macro_f1": res["macro_f1"], "n": int(len(df)),
                       "model_version": clf.config["version"], "model_created_at": clf.config["created_at"]}, f, indent=2)
        print(f"\nLock dibuat: {lock}")


if __name__ == "__main__":
    main()
