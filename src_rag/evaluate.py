from pathlib import Path
from time import sleep
import os
import re

import mlflow
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import yaml

from NLP_groupe6.src_rag import models

ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = ROOT / "config.yml"
CONF = yaml.safe_load(open(CONF_PATH, encoding="utf-8")) if CONF_PATH.exists() else {}

# ✅ Corpus : tous les .txt (uniformisé en str pour éviter Path/str mix)
FOLDER = ROOT / "wikipedia_pages"
FILENAMES = [str(p) for p in sorted(FOLDER.glob("*.txt"))]

# ✅ Questions
DF = pd.read_csv(ROOT / "data/raw/questions.csv", sep=";", encoding="utf-8")

# ✅ Similarité sémantique FR-friendly
ENCODER = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device="cpu",
)

# ✅ Pacing configurable
EVAL_SLEEP_SEC = float(os.getenv("EVAL_SLEEP_SEC", "0.2"))

# ✅ Nom d'expérience configurable (et pas “figé” à l'import)
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "RAG_Civilisations")


def _load_ml_flow():
    """
    Évite les effets de bord bizarres : on garde un nom d'expérience configurable.
    run_experiments.py peut aussi appeler mlflow.set_experiment(...) ensuite.
    """
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except Exception:
        pass


_load_ml_flow()


# -------------------------
# Helpers: sécurité + robustesse
# -------------------------
def redact_secrets(obj):
    """
    Retire récursivement tout ce qui ressemble à une clé/secret dans les configs loggées.
    """
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            ks = str(k).lower()
            if any(x in ks for x in ["key", "token", "secret", "password", "api_key", "apikey"]):
                clean[k] = "***REDACTED***"
            else:
                clean[k] = redact_secrets(v)
        return clean
    if isinstance(obj, list):
        return [redact_secrets(x) for x in obj]
    return obj


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def safe_reply(rag, question: str, max_retries: int = 5) -> str:
    """
    Retry simple avec backoff. Stoppe net si 'decommissioned'.
    """
    delay = 1.0
    last_err = None

    for _ in range(max_retries):
        try:
            return rag.reply(question)
        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # modèle décommissionné => inutile de retry
            if "decommissioned" in msg or "model_decommissioned" in msg:
                raise

            # erreurs transitoires fréquentes
            if any(x in msg for x in ["rate limit", "429", "timeout", "temporarily", "503", "connection"]):
                sleep(delay)
                delay = min(delay * 2, 8.0)
                continue

            # autres erreurs => on remonte direct
            raise

    return f"[ERROR] Failed after {max_retries} retries: {last_err}"


# -------------------------
# Runner
# -------------------------
def run_evaluate_retrieval(config, rag=None):
    rag = rag or models.get_model(config)
    score = evaluate_retrieval(rag, FILENAMES, DF.dropna().copy())
    description = str(config.get("model", {})) if isinstance(config, dict) else None
    _push_mlflow_result(score, config, description)
    return rag


def run_evaluate_reply(config, rag=None):
    rag = rag or models.get_model(config)
    df_eval = DF.dropna().copy()
    score = evaluate_reply(rag, FILENAMES, df_eval)
    description = str(config.get("model", {})) if isinstance(config, dict) else None
    _push_mlflow_result(score, config, description)
    return rag


def _push_mlflow_result(score, config, description=None):
    """
    Compat MLflow : certaines versions ne supportent pas start_run(description=...).
    On fallback vers run_name + tag.
    """
    df = score.pop("df_result", None)

    try:
        # MLflow récent (parfois OK)
        with mlflow.start_run(description=description):
            _log_all(score, df, config, description)
    except TypeError:
        # MLflow ancien => fallback
        with mlflow.start_run(run_name=(description[:120] if description else None)):
            if description:
                mlflow.set_tag("description", description)
            _log_all(score, df, config, description)
    except Exception:
        # ultra safe fallback
        with mlflow.start_run():
            if description:
                mlflow.set_tag("description", description)
            _log_all(score, df, config, description)


def _log_all(score, df, config, description=None):
    # metrics
    try:
        mlflow.log_metrics(score)
    except Exception:
        for k, v in score.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

    # table / artifact
    if df is not None:
        try:
            mlflow.log_table(df, artifact_file="df.json")
        except Exception:
            try:
                # fallback csv
                tmp = Path("df_result.csv")
                df.to_csv(tmp, index=False, encoding="utf-8")
                mlflow.log_artifact(str(tmp))
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # config (redacted)
    try:
        mlflow.log_dict(redact_secrets(config), "config.json")
    except Exception:
        pass


# -------------------------
# Metrics
# -------------------------
def evaluate_reply(rag, filenames, df):
    rag.load_files(filenames)

    replies = []
    for question in tqdm(df["question"], desc="Reply eval"):
        replies.append(safe_reply(rag, question))
        sleep(EVAL_SLEEP_SEC)

    df.loc[:, "reply"] = replies
    df.loc[:, "sim"] = df.apply(
        lambda row: calc_semantic_similarity(str(row["reply"]), str(row["expected_reply"])),
        axis=1,
    )
    df.loc[:, "is_correct"] = df["sim"] > 0.7

    return {
        "reply_similarity": float(df["sim"].mean()),
        "percent_correct": float(df["is_correct"].mean()),
        "df_result": df[["question", "reply", "expected_reply", "sim", "is_correct"]],
    }


def evaluate_retrieval(rag, filenames, df_question):
    rag.load_files(filenames)

    ranks = []
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0

    for _, row in df_question.iterrows():
        chunks = rag._get_context(row["question"])  # OK mais méthode privée

        gold = normalize_text(row.get("text_answering", ""))
        found_rank = 0

        if gold:
            for i, c in enumerate(chunks, start=1):
                if gold in normalize_text(c):
                    found_rank = i
                    break

        ranks.append(found_rank)

        if found_rank == 1:
            hit_at_1 += 1
        if 1 <= found_rank <= 3:
            hit_at_3 += 1
        if 1 <= found_rank <= 5:
            hit_at_5 += 1

    df_question.loc[:, "rank"] = ranks

    mrr = float(np.mean([0.0 if r == 0 else 1.0 / r for r in ranks]))
    n = len(ranks) if len(ranks) else 1

    return {
        "mrr": mrr,
        "hit@1": float(hit_at_1 / n),
        "hit@3": float(hit_at_3 / n),
        "hit@5": float(hit_at_5 / n),
        "nb_chunks": int(len(rag.get_chunks())),
        "df_result": df_question[["question", "text_answering", "rank"]],
    }


def calc_semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    if not generated_answer and not reference_answer:
        return 1.0
    embeddings = ENCODER.encode([generated_answer, reference_answer], normalize_embeddings=True)
    # embeddings normalisés => cos = dot
    return float(np.dot(embeddings[0], embeddings[1]))
