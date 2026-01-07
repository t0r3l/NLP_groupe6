# run_experiments.py
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Tuple
import re
import mlflow
from src_rag import evaluate as eval_mod
from src_rag import models


# -----------------------------
# (Optionnel) Ping Groq (ne doit PAS bloquer le script)
# -----------------------------
def _optional_ping_groq() -> None:
    """
    Petit test réseau/API pour vérifier que la clé fonctionne.
    Ne doit JAMAIS faire crasher run_experiments.
    """
    try:
        from openai import OpenAI

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            print("⚠️ GROQ_API_KEY absent -> ping skipped")
            return

        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

        # Modèle supporté (remplace llama3-8b-8192 décommissionné)
        ping_model = os.getenv("GROQ_PING_MODEL", "llama-3.1-8b-instant")

        resp = client.chat.completions.create(
            model=ping_model,
            messages=[{"role": "user", "content": "ping"}],
        )
        print(f"✅ Groq ping OK ({ping_model}) ->", resp.choices[0].message.content)
    except Exception as e:
        print("⚠️ Groq ping failed (non-bloquant):", e)


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


def _mlflow_safe_name(name: str) -> str:
    # Ex: hit@1 -> hit_at_1 (plus lisible que hit_1)
    name = name.replace("@", "_at_")
    # Remplacer tout caractère interdit par "_"
    return re.sub(r"[^0-9A-Za-z_\-\. /]", "_", name)


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
                mlflow.log_metric(_mlflow_safe_name(k), float(v))

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
    score = eval_mod.evaluate_reply(rag, eval_mod.FILENAMES, eval_mod.DF.dropna().copy())
    _log_run(stage="reply", score=score, config=config)
    return float(score["reply_similarity"]), float(score["percent_correct"])


def _overlaps_for(chunk_size: int) -> List[int]:
    cand = {
        0,
        max(0, chunk_size // 16),
        max(0, chunk_size // 8),
        min(64, max(0, chunk_size // 4)),
    }
    return sorted([x for x in cand if x < chunk_size])


# -----------------------------
# MAIN
# -----------------------------
def main():
    # (Optionnel) test clé + API (non bloquant)
    _optional_ping_groq()
    print("MLFLOW_TRACKING_URI =", mlflow.get_tracking_uri())

    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "RAG_Civilisations_MRR")
    mlflow.set_experiment(experiment_name)

    embeddings = [
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
    ]

    include_heavy = os.getenv("INCLUDE_HEAVY", "0") == "1"
    if include_heavy:
        embeddings += ["BAAI/bge-m3"]

    chunk_sizes = [192, 256, 384]
    small2big_opts = [False, True]

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

    results.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    top_k_refine = int(os.getenv("TOPK_REFINE", "2"))
    top_k_reply = int(os.getenv("TOPK_REPLY", "3"))

    best_coarse = results[:top_k_refine]
    print("\n[TOP COARSE]")
    for rank, (mrr, nb_chunks, cfg) in enumerate(best_coarse, start=1):
        print(f"  {rank}) mrr={mrr:.4f} nb_chunks={nb_chunks} cfg={cfg}")

    refine_candidates: List[Dict[str, Any]] = []
    for _, _, base_cfg in best_coarse:
        cs = base_cfg["chunk_size"]
        emb = base_cfg["embedding"]
        for ov in _overlaps_for(cs):
            for s2b in small2big_opts:
                refine_candidates.append(
                    {"embedding": emb, "chunk_size": cs, "overlap": ov, "small2big": s2b}
                )

    uniq, seen = [], set()
    for c in refine_candidates:
        key = (c["embedding"], c["chunk_size"], c["overlap"], c["small2big"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    refine_candidates = uniq

    refine_results: List[Tuple[float, int, Dict[str, Any]]] = []
    print(f"\n[PHASE 2] Refine MRR: {len(refine_candidates)} runs")
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
