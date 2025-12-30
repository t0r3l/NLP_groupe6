from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from time import sleep
import yaml

from src_rag import models

ROOT = Path(__file__).resolve().parents[1]
CONF = yaml.safe_load(open(ROOT / "config.yml", encoding="utf-8"))

# ✅ Corpus : tous les .txt
FOLDER = ROOT / "wikipedia_pages"
FILENAMES = sorted(FOLDER.glob("*.txt"))

# ✅ Questions
DF = pd.read_csv(ROOT / "data/raw/questions.csv", sep=";", encoding="utf-8")

# Pour mesurer la similarité sémantique des réponses (évaluation)
ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def _load_ml_flow(conf):
    mlflow.set_experiment("RAG_Civilisations")


_load_ml_flow(CONF)


def run_evaluate_retrieval(config, rag=None):
    rag = rag or models.get_model(config)
    score = evaluate_retrieval(rag, FILENAMES, DF.dropna().copy())
    description = str(config.get("model", {}))
    _push_mlflow_result(score, config, description)
    return rag


def run_evaluate_reply(config, rag=None):
    rag = rag or models.get_model(config)

    # ✅ Comme DF est petit (~10), on les prend toutes
    df_eval = DF.dropna().copy()

    score = evaluate_reply(rag, FILENAMES, df_eval)
    description = str(config.get("model", {}))
    _push_mlflow_result(score, config, description)
    return rag


def _push_mlflow_result(score, config, description=None):
    with mlflow.start_run(description=description):
        df = score.pop("df_result")
        mlflow.log_table(df, artifact_file="df.json")
        mlflow.log_metrics(score)

        config_no_key = {k: v for k, v in config.items() if not str(k).endswith("_key")}
        mlflow.log_dict(config_no_key, "config.json")


def evaluate_reply(rag, filenames, df):
    rag.load_files(filenames)

    replies = []
    for question in tqdm(df["question"], desc="Reply eval"):
        replies.append(rag.reply(question))
        sleep(1)  # un peu moins long, mais safe pour Groq

    df.loc[:, "reply"] = replies
    df.loc[:, "sim"] = df.apply(
        lambda row: calc_semantic_similarity(row["reply"], row["expected_reply"]),
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
    for _, row in df_question.iterrows():
        chunks = rag._get_context(row["question"])

        # ✅ FIX IMPORTANT :
        # rank doit être 1..k (et 0 si pas trouvé), sinon le top-1 est compté comme 0 !
        try:
            rank = 1 + next(i for i, c in enumerate(chunks) if row["text_answering"] in c)
        except StopIteration:
            rank = 0

        ranks.append(rank)

    df_question.loc[:, "rank"] = ranks

    mrr = float(np.mean([0.0 if r == 0 else 1.0 / r for r in ranks]))

    return {
        "mrr": mrr,
        "nb_chunks": int(len(rag.get_chunks())),
        "df_result": df_question[["question", "text_answering", "rank"]],
    }


def calc_semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    embeddings = ENCODER.encode([generated_answer, reference_answer])
    sim = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    return float(sim)
