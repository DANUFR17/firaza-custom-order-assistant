"""
eval_rag.py - Evaluasi retrieval RAG (terukur): hit@1, hit@3, MRR untuk pertanyaan dalam domain, dan kemampuan menolak
pertanyaan di luar domain. Juga menghitung ambang skor (RAG_MIN_SCORE) yang disarankan.

Jalankan dari root proyek (Ollama berjalan dan bge-m3 sudah di-pull):
    python src/eval_rag.py
    python src/eval_rag.py --generate 3     # sekaligus coba menjawab 3 pertanyaan (butuh model bahasa)
Hasil: reports/rag_eval.json
"""
import argparse
import csv
import json
import os

import numpy as np

from rag_service import RagService, RagUnavailable


def evaluate(svc, eval_path="data/rag_eval.csv"):
    rows = list(csv.DictReader(open(eval_path, encoding="utf-8")))
    inn, ood = [], []
    for r in rows:
        hits = svc.retrieve(r["question"], k=5)
        top = hits[0]["skor"]
        if r["in_domain"] == "1":
            exp = set(r["expected_ids"].split("|"))
            rank = next((i for i, h in enumerate(hits, 1) if h["id"] in exp), None)
            inn.append({"q": r["question"], "expected": sorted(exp), "rank": rank, "top_id": hits[0]["id"], "top_score": top})
        else:
            ood.append({"q": r["question"], "top_id": hits[0]["id"], "top_score": top})
    ranks = [x["rank"] for x in inn]
    in_scores, ood_scores = [x["top_score"] for x in inn], [x["top_score"] for x in ood]
    res = {
        "embedding": svc.embedder.name, "indeks": svc.index.backend, "n_dalam_domain": len(inn), "n_luar_domain": len(ood),
        "hit@1": round(sum(1 for r in ranks if r == 1) / len(inn), 3),
        "hit@3": round(sum(1 for r in ranks if r and r <= 3) / len(inn), 3),
        "mrr": round(float(np.mean([1 / r if r else 0 for r in ranks])), 3),
        "skor_teratas_dalam_domain": {"min": round(min(in_scores), 3), "rata2": round(float(np.mean(in_scores)), 3)},
        "skor_teratas_luar_domain": {"maks": round(max(ood_scores), 3), "rata2": round(float(np.mean(ood_scores)), 3)},
        "ambang_saat_ini": svc.min_score,
        "ditolak_benar_luar_domain": round(sum(1 for s in ood_scores if s < svc.min_score) / len(ood), 3),
        "ditolak_salah_dalam_domain": round(sum(1 for s in in_scores if s < svc.min_score) / len(inn), 3),
        "ambang_disarankan": round((max(ood_scores) + min(in_scores)) / 2, 3) if max(ood_scores) < min(in_scores) else None,
        "gagal_hit@3": [{"pertanyaan": x["q"], "diharapkan": x["expected"], "teratas": x["top_id"]}
                        for x in inn if not x["rank"] or x["rank"] > 3],
    }
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/rag_eval.csv")
    ap.add_argument("--out", default="reports/rag_eval.json")
    ap.add_argument("--generate", type=int, default=0)
    a = ap.parse_args()
    svc = RagService()
    try:
        res = evaluate(svc, a.eval)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        if res["ambang_disarankan"] is None:
            print("\nCatatan: skor dalam domain dan luar domain tumpang tindih; ambang tunggal tidak memisahkan keduanya sempurna.")
        else:
            print(f"\nAmbang disarankan: set RAG_MIN_SCORE={res['ambang_disarankan']} di .env bila berbeda dari ambang saat ini.")
        for r in list(csv.DictReader(open(a.eval, encoding="utf-8")))[: a.generate]:
            out = svc.ask(r["question"])
            print(f"\nQ: {r['question']}\nA ({out['mode']}): {out['jawaban']}\nSumber: {[s['id'] for s in out['sumber']]}")
    except RagUnavailable as e:
        raise SystemExit(f"GAGAL: {e}")
