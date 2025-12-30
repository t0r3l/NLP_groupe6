# run_experiments.py
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Tuple

import mlflow
from src_rag import evaluate as eval_mod
from src_rag import models


# -----------------------------
# Helpers
# -----------------------------
def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _log_run(stage: str, score: Dict[str, Any], config: Dict[str, Any]) -> None:
    score = dict(score)
    df = score.pop("df_result", None)

    config_no_key = {k: v for k, v in config.items() if not str(k).endswith("_key")}
    flat_params = _flatten_dict(config_no_key)

    run_name = (
        f"{stage} | {flat_params.get('model.embedding', '?')} | "
        f"cs={flat_params.get('model.chunk_size', '?')} "
        f"ov={flat_params.get('model.overlap', '?')} "
        f"s2b={flat_params.get('model.small2big', '?')}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", stage)

        for k, v in flat_params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                mlflow.log_param(k, str(v))

        for k, v in score.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        if df is not None:
            try:
                mlflow.log_table(df, artifact_file=f"{stage}_df.json")
            except Exception:
                pass

        mlflow.log_dict(config_no_key, f"{stage}_config.json")


def _run_retrieval_once(model_cfg: Dict[str, Any]) -> Tuple[float, int]:
    config = {"model": model_cfg}
    rag = models.get_model(config)
    score = eval_mod.evaluate_retrieval(rag, eval_mod.FILENAMES, eval_mod.DF.dropna().copy())
    _log_run(stage="retrieval", score=score, config=config)
    return float(score["mrr"]), int(score["nb_chunks"])


def _run_reply_once(model_cfg: Dict[str, Any]) -> Tuple[float, float]:
    config = {"model": model_cfg}
    rag = models.get_model(config)

    # DF petit (~10) => on évalue tout
    score = eval_mod.evaluate_reply(rag, eval_mod.FILENAMES, eval_mod.DF.dropna().copy())

    _log_run(stage="reply", score=score, config=config)
    return float(score["reply_similarity"]), float(score["percent_correct"])


def _overlaps_for(chunk_size: int) -> List[int]:
    # overlap capé à 64 pour ne pas exploser nb_chunks
    cand = {
        0,
        max(0, chunk_size // 16),
        max(0, chunk_size // 8),
        min(64, max(0, chunk_size // 4)),
    }
    # garder uniquement < chunk_size
    return sorted([x for x in cand if x < chunk_size])


# -----------------------------
# MAIN
# -----------------------------
def main():
    # IMPORTANT : lance MLflow UI depuis la racine du projet
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "RAG_Civilisations_MRR")
    mlflow.set_experiment(experiment_name)

    # ---------- Embeddings (priorité FR / multilingual)
    # On passe des noms HF COMPLETS : ton models.py accepte ça (fallback).
    embeddings = [
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # souvent top en FR
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # plus léger
        "sentence-transformers/distiluse-base-multilingual-cased-v2",  # rapide / solide
    ]

    # Optionnel : activer un embedding plus lourd (si tu veux tester)
    # set INCLUDE_HEAVY=1
    include_heavy = os.getenv("INCLUDE_HEAVY", "0") == "1"
    if include_heavy:
        embeddings += [
            "BAAI/bge-m3",  # multilingue retrieval, plus lourd -> peut être plus lent en CPU
        ]

    # ---------- Chunk sizes (bons candidats pour MRR sur wiki)
    # (petit/moyen = souvent mieux pour retrouver précisément text_answering)
    chunk_sizes = [192, 256, 384]

    # ---------- small2big : on teste ON/OFF
    small2big_opts = [False, True]

    # ---------- PHASE 1 : COARSE (MRR d’abord)
    # overlap fixé = chunk_size//8 (bon point de départ)
    coarse_candidates: List[Dict[str, Any]] = []
    for emb in embeddings:
        for cs in chunk_sizes:
            ov = max(0, cs // 8)
            for s2b in small2big_opts:
                coarse_candidates.append(
                    {"embedding": emb, "chunk_size": cs, "overlap": ov, "small2big": s2b}
                )

    results: List[Tuple[float, int, Dict[str, Any]]] = []
    print(f"\n[PHASE 1] Coarse MRR search: {len(coarse_candidates)} runs")
    for i, cfg in enumerate(coarse_candidates, start=1):
        try:
            print(f"  - ({i}/{len(coarse_candidates)}) {cfg}")
            mrr, nb_chunks = _run_retrieval_once(cfg)
            results.append((mrr, nb_chunks, cfg))
            print(f"    -> mrr={mrr:.4f} | nb_chunks={nb_chunks}")
        except Exception as e:
            print(f"    !! FAILED: {e}")
            traceback.print_exc()

    if not results:
        print("\nAucun run retrieval n'a réussi. Vérifie evaluate.py (paths) + dépendances.")
        return

    # Trie : MRR décroissant, et si égalité -> moins de chunks (souvent mieux)
    results.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    top_k_refine = int(os.getenv("TOPK_REFINE", "2"))  # ✅ par défaut : 2 pour limiter le coût
    top_k_reply = int(os.getenv("TOPK_REPLY", "3"))  # reply uniquement sur top 3

    best_coarse = results[:top_k_refine]
    print("\n[TOP COARSE]")
    for rank, (mrr, nb_chunks, cfg) in enumerate(best_coarse, start=1):
        print(f"  {rank}) mrr={mrr:.4f} nb_chunks={nb_chunks} cfg={cfg}")

    # ---------- PHASE 2 : REFINE (MRR)
    refine_candidates: List[Dict[str, Any]] = []
    for _, _, base_cfg in best_coarse:
        cs = base_cfg["chunk_size"]
        emb = base_cfg["embedding"]
        for ov in _overlaps_for(cs):
            for s2b in small2big_opts:
                refine_candidates.append(
                    {"embedding": emb, "chunk_size": cs, "overlap": ov, "small2big": s2b}
                )

    # dédoublonnage
    uniq, seen = [], set()
    for c in refine_candidates:
        key = (c["embedding"], c["chunk_size"], c["overlap"], c["small2big"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    refine_candidates = uniq

    refine_results: List[Tuple[float, int, Dict[str, Any]]] = []
    print(f"\n[PHASE 2] Refine MRR: {len(refine_candidates)} runs (around TOP {top_k_refine})")
    for i, cfg in enumerate(refine_candidates, start=1):
        try:
            print(f"  - ({i}/{len(refine_candidates)}) {cfg}")
            mrr, nb_chunks = _run_retrieval_once(cfg)
            refine_results.append((mrr, nb_chunks, cfg))
            print(f"    -> mrr={mrr:.4f} | nb_chunks={nb_chunks}")
        except Exception as e:
            print(f"    !! FAILED: {e}")
            traceback.print_exc()

    all_results = results + refine_results
    all_results.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    print("\n[TOP FINAL RETRIEVAL]")
    for rank, (mrr, nb_chunks, cfg) in enumerate(all_results[:10], start=1):
        print(f"  {rank}) mrr={mrr:.4f} nb_chunks={nb_chunks} cfg={cfg}")

    # ---------- PHASE 3 : REPLY (seulement sur les meilleurs)
    print(f"\n[PHASE 3] Reply eval on TOP {top_k_reply} configs")
    for i, (mrr, nb_chunks, cfg) in enumerate(all_results[:top_k_reply], start=1):
        try:
            print(f"  - ({i}/{top_k_reply}) cfg={cfg} (mrr={mrr:.4f})")
            rep_sim, pct_ok = _run_reply_once(cfg)
            print(f"    -> reply_similarity={rep_sim:.4f} | percent_correct={pct_ok:.4f}")
        except Exception as e:
            print(f"    !! FAILED reply: {e}")
            traceback.print_exc()

    print("\n✅ Terminé. Ouvre MLflow :")
    print("   cd <racine-projet>")
    print("   python -m mlflow ui")
    print("   -> http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
