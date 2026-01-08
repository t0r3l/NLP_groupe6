"""
Evaluation utilities for RAG system.
Shared functions and evaluation metrics.
"""
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep
import yaml

from src_rag import models
from src_rag.models import calc_semantic_similarity

ROOT = Path(__file__).resolve().parents[1]
CONF = yaml.safe_load(open(ROOT / "config.yml"))

# Wikipedia data paths
WIKI_FOLDER = ROOT / "data" / "raw" / "wikipedia_pages"
QUESTIONS_PATH = ROOT / "data" / "raw" / "questions.csv"


def get_wiki_files():
    """Get list of Wikipedia files."""
    return list(WIKI_FOLDER.glob("*.txt"))


def get_questions_df():
    """Load questions dataframe."""
    return pd.read_csv(QUESTIONS_PATH, sep=";").dropna()


def _load_ml_flow():
    mlflow.set_experiment("RAG_historian_wikipedia")


def run_evaluate_retrieval(config, rag=None):
    rag = rag or models.get_model(config)
    filenames = get_wiki_files()
    df = get_questions_df()
    score = evaluate_retrieval(rag, filenames, df)

    description = str(config.get("model", {}))
    _push_mlflow_result(score, config, description)

    return rag


def run_evaluate_reply(config, rag=None):
    rag = rag or models.get_model(config)
    filenames = get_wiki_files()
    df = get_questions_df()
    indexes = range(2, len(df), 10)
    score = evaluate_reply(rag, filenames, df.iloc[indexes])

    description = str(config.get("model", {}))
    _push_mlflow_result(score, config, description)

    return rag


def _push_mlflow_result(score, config, description=None):
    with mlflow.start_run(description=description):
        df = score.pop("df_result")
        mlflow.log_table(df, artifact_file="df.json")
        mlflow.log_metrics(score)

        config_no_key = {
            key: val for key, val in config.items() if not key.endswith("_key")
        }

        mlflow.log_dict(config_no_key, "config.json")


def evaluate_reply(rag, filenames, df):
    rag.load_wikipedia_files(filenames)

    replies = []
    for question in tqdm(df["question"]):
        replies.append(rag.reply(question))
        # Not too many requests to groq
        sleep(2)

    df["reply"] = replies
    df["sim"] = df.apply(
        lambda row: calc_semantic_similarity(row["reply"], row["expected_reply"]),
        axis=1,
    )
    df["is_correct"] = df["sim"] > 0.7

    return {
        "reply_similarity": df["sim"].mean(),
        "percent_correct": df["is_correct"].mean(),
        "df_result": df[["question", "reply", "expected_reply", "sim", "is_correct"]],
    }


def evaluate_retrieval(rag, filenames, df_question):
    rag.load_wikipedia_files(filenames)
    ranks = []
    for _, row in df_question.iterrows():
        # For each question, get the 5 most relevant chunks
        chunks = rag._get_context(row.question)
        try:
            # Find the rank of the chunk containing the answer based on text_answering column
            rank = next(i for i, c in enumerate(chunks) if row.text_answering in c)
        except StopIteration:
            rank = 0
        ranks.append(rank)

    df_question["rank"] = ranks
    mrr = np.mean([0 if r == 0 else 1 / r for r in ranks])

    return {
        "mrr": mrr,
        "nb_chunks": len(rag.get_chunks()),
        "df_result": df_question[["question", "text_answering", "rank"]]
    }


def calc_acceptable_chunks(chunks, text_to_find):
    acceptable_chunks = []
    for answer in text_to_find:
        chunks_ok = set[int](i for i, chunk in enumerate(chunks) if answer in chunk)
        acceptable_chunks.append(chunks_ok)

    return acceptable_chunks


def calc_mrr(sim_score, acceptable_chunks, top_n=5):
    ranks = []
    for this_score, this_acceptable_chunks in zip(sim_score, acceptable_chunks):
        indexes = reversed(np.argsort(this_score))
        try:
            rank = 1 + next(i for i, idx in enumerate(indexes) if idx in this_acceptable_chunks)
        except StopIteration:
            rank = len(this_score) + 1

        ranks.append(rank)

    return {
        "mrr": sum(1 / r if r < top_n + 1 else 0 for r in ranks) / len(ranks),
        "ranks": ranks,
    }


# calc_semantic_similarity is imported from models.py at the top


if __name__ == "__main__":
    _load_ml_flow()
    model_config = {"chunk_size": 256, "small2big": True, "add_metadata": True}
    # run_evaluate_retrieval({"model": model_config})
    run_evaluate_reply({"model": model_config})
