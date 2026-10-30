"""
train_keras.py - Model klasifikasi order Firaza yang dilatih DARI NOL dengan TensorFlow/Keras.

Jalankan dari root proyek:
    python src/train_keras.py --data-dir data --out-dir reports --model-dir models
    python src/train_keras.py --arch cnn --seeds 1 --epochs 5      # uji cepat
    python src/train_keras.py --augment                            # + augmentasi token

Arsitektur (tanpa model pra-latih, tanpa TF Hub, tanpa AutoML):
    teks ter-preprocess -> TextVectorization -> Embedding(64) ->
      (a) cnn   : Conv1D kernel 2/3/4 + GlobalMaxPool, atau
      (b) bilstm: Bidirectional LSTM
    -> Dense(64) -> Dropout -> Softmax(5 kelas)

Protokol:
 - Dilatih di `train`, early stopping dan pemilihan arsitektur/seed HANYA dari `val`.
 - Dilaporkan: val, test (grouped), manual. Tiap arsitektur dilatih dengan beberapa seed
   dan dilaporkan mean +- std, karena data kecil membuat satu seed tidak stabil.
 - `manual_test_blind.csv` tidak disentuh (lihat evaluate_blind.py).

Catatan inferensi: model ini menerima teks yang SUDAH lewat preprocessing.preprocess().
TextVectorization ikut tersimpan di dalam file .keras, tetapi preprocess() dipanggil
terpisah (di API) sebelum model.
"""
import argparse
import os
import random
import time

# Sembunyikan log info C++ TensorFlow (oneDNN, cpu_feature_guard). Harus di atas `import tensorflow`.
# Mematikan oneDNN bisa menggeser angka di desimal ketiga dibanding run sebelumnya (hanya
# pembulatan floating-point); kalau mau angka persis sama dengan run lama, hapus baris kedua.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from common import LABEL2ID, LABELS, evaluate, load_data, misclassified, plot_confusion, prep, save_json

MAX_TOKENS = 4000
SEQ_LEN = 40
EMB_DIM = 64


def build_model(arch, train_texts):
    tv = layers.TextVectorization(max_tokens=MAX_TOKENS, standardize=None, split="whitespace",
                                  output_mode="int", output_sequence_length=SEQ_LEN, name="vectorizer")
    tv.adapt(tf.constant(train_texts))

    inp = layers.Input(shape=(), dtype=tf.string, name="text")
    x = tv(inp)
    if arch == "cnn":
        x = layers.Embedding(tv.vocabulary_size(), EMB_DIM, name="embedding")(x)
        x = layers.SpatialDropout1D(0.2)(x)
        pooled = [layers.GlobalMaxPooling1D()(layers.Conv1D(64, k, padding="same", activation="relu")(x))
                  for k in (2, 3, 4)]
        x = layers.Dropout(0.4)(layers.concatenate(pooled))
    elif arch == "bilstm":
        x = layers.Embedding(tv.vocabulary_size(), EMB_DIM, mask_zero=True, name="embedding")(x)
        x = layers.SpatialDropout1D(0.2)(x)
        x = layers.Bidirectional(layers.LSTM(48, dropout=0.2))(x)
        x = layers.Dropout(0.4)(x)
    else:
        raise ValueError(arch)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(LABELS), activation="softmax", name="label")(x)
    model = models.Model(inp, out, name=f"firaza_{arch}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def augment_text(text, rng, p_unk=0.12, p_del=0.05):
    """Augmentasi sederhana: acak ganti kata dengan token asing (jadi [UNK]) atau hapus kata.
    Melatih model agar tidak bergantung pada satu kata dan tahan terhadap kosakata baru."""
    out = []
    for w in text.split():
        r = rng.random()
        if r < p_del:
            continue
        out.append("xunk" if r < p_del + p_unk else w)
    return " ".join(out) if out else text


def predict_labels(model, texts):
    probs = model.predict(tf.constant(texts), verbose=0, batch_size=128)
    return np.array([LABELS[i] for i in probs.argmax(1)])


def plot_history(history, title, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, key, name in ((ax[0], "loss", "Loss"), (ax[1], "accuracy", "Accuracy")):
        a.plot(history.history[key], label="train")
        a.plot(history.history["val_" + key], label="val")
        a.set_title(f"{title} - {name}")
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dataset", default=None, help="nama file dataset (default: dataset_sintetis_v3.csv)")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--arch", default="both", choices=["cnn", "bilstm", "both"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--augment", action="store_true", help="tambah 1 salinan teks train yang diaugmentasi")
    args = ap.parse_args()

    t0 = time.time()
    _, sp = load_data(args.data_dir, args.dataset)
    X = {k: prep(v.text) for k, v in sp.items()}
    y = {k: v.label.map(LABEL2ID).values for k, v in sp.items()}
    archs = ["cnn", "bilstm"] if args.arch == "both" else [args.arch]
    train_x, train_y = X["train"], y["train"]
    if args.augment:
        rng_aug = random.Random(0)
        train_x = X["train"] + [augment_text(t, rng_aug) for t in X["train"]]
        train_y = np.concatenate([y["train"], y["train"]])

    runs = {a: [] for a in archs}
    for arch in archs:
        for seed in range(args.seeds):
            tf.keras.utils.set_random_seed(42 + seed)
            model = build_model(arch, X["train"])
            hist = model.fit(
                tf.constant(train_x), train_y, epochs=args.epochs, batch_size=args.batch_size,
                validation_data=(tf.constant(X["val"]), y["val"]), verbose=0,
                callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
            res = {"seed": 42 + seed, "epochs_run": len(hist.history["loss"])}
            preds = {}
            for split in ("val", "test", "manual"):
                preds[split] = predict_labels(model, X[split])
                res[split] = evaluate(sp[split].label.values, preds[split])
            res["_model"], res["_hist"], res["_preds"] = model, hist, preds
            runs[arch].append(res)
            print(f"{arch:7s} seed {42 + seed}: ep={res['epochs_run']:2d}  "
                  f"val={res['val']['macro_f1']:.3f}  test={res['test']['macro_f1']:.3f}  "
                  f"manual={res['manual']['macro_f1']:.3f}")

    # --- ringkasan mean +- std per arsitektur ---
    summary = {}
    for arch, rs in runs.items():
        summary[arch] = {}
        for split in ("val", "test", "manual"):
            f1s = [r[split]["macro_f1"] for r in rs]
            summary[arch][split] = {"mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s)), 4)}
    print("\nRingkasan macro F1 (mean +- std antar seed):")
    for arch, s in summary.items():
        print(f"  {arch:7s} " + "  ".join(f"{k}={v['mean']:.3f}+-{v['std']:.3f}" for k, v in s.items()))

    # --- pilih arsitektur dan seed HANYA dari val ---
    best_arch = max(summary, key=lambda a: summary[a]["val"]["mean"])
    best = max(runs[best_arch], key=lambda r: r["val"]["macro_f1"])
    print(f"\nArsitektur terpilih (val): {best_arch}, seed {best['seed']}")

    os.makedirs(args.model_dir, exist_ok=True)
    best["_model"].save(os.path.join(args.model_dir, f"keras_{best_arch}.keras"))
    plot_history(best["_hist"], best_arch, os.path.join(args.out_dir, "keras_training_curve.png"))
    for split in ("val", "test", "manual"):
        plot_confusion(sp[split].label.values, best["_preds"][split], f"Keras {best_arch} - {split}",
                       os.path.join(args.out_dir, f"keras_cm_{split}.png"))
        misclassified(sp[split], best["_preds"][split]).to_csv(
            os.path.join(args.out_dir, f"keras_errors_{split}.csv"), index=False)
    save_json({
        "selected": {"arch": best_arch, "seed": best["seed"], "selection": "val macro F1 (mean antar seed)"},
        "hyperparameters": {"max_tokens": MAX_TOKENS, "seq_len": SEQ_LEN, "emb_dim": EMB_DIM,
                            "batch_size": args.batch_size, "optimizer": "Adam(1e-3)", "early_stopping": "val_loss, patience 5"},
        "summary_mean_std_over_seeds": summary,
        "selected_model_metrics": {s: best[s] for s in ("val", "test", "manual")},
    }, os.path.join(args.out_dir, "keras_metrics.json"))
    print(f"Selesai dalam {time.time() - t0:.0f} detik. Model: {args.model_dir}/keras_{best_arch}.keras")


if __name__ == "__main__":
    main()
