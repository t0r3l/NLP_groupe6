"""
Evaluation script for Wikipedia-based RAG on African civilizations.
Tests different embeddings, chunk sizes, overlap, small2big, and metadata.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from time import sleep
import yaml
import json
from datetime import datetime

from src_rag import models

ROOT = Path(__file__).resolve().parents[1]
CONF = yaml.safe_load(open(ROOT / "config.yml"))

# Wikipedia files
WIKI_FOLDER = ROOT / "data" / "raw" / "wikipedia_pages"
WIKI_FILES = list(WIKI_FOLDER.glob("*.txt"))

# Questions CSV
DF = pd.read_csv(ROOT / "data/raw/questions.csv", sep=";").dropna()

# Encoder for semantic similarity
ENCODER = SentenceTransformer("all-MiniLM-L6-v2")


def calc_semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    """Calculate semantic similarity between generated and reference answers."""
    embeddings = ENCODER.encode([generated_answer, reference_answer])
    generated_embedding = embeddings[0].reshape(1, -1)
    reference_embedding = embeddings[1].reshape(1, -1)
    similarity = cosine_similarity(generated_embedding, reference_embedding)[0][0]
    return float(similarity)


def evaluate_retrieval(rag, df_question):
    """Evaluate retrieval quality using MRR."""
    ranks = []
    for _, row in df_question.iterrows():
        chunks = rag._get_context(row.question)
        try:
            rank = next(i + 1 for i, c in enumerate(chunks) if row.text_answering in c)
        except StopIteration:
            rank = 0
        ranks.append(rank)

    mrr = np.mean([0 if r == 0 else 1 / r for r in ranks])
    return {
        "mrr": mrr,
        "nb_chunks": len(rag.get_chunks()),
        "ranks": ranks,
    }


def evaluate_reply(rag, df_question, sample_step=1):
    """Evaluate reply quality using semantic similarity."""
    df = df_question.copy()
    
    replies = []
    for question in tqdm(df["question"], desc="Generating replies"):
        replies.append(rag.reply(question))
        sleep(2)  # Rate limiting for Groq API

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


def run_test(config: dict, run_reply: bool = False, sample_step: int = 1) -> dict:
    """
    Run a single test configuration.
    
    Args:
        config: RAG configuration (chunk_size, overlap, small2big, embedding, add_metadata)
        run_reply: If True, also run reply evaluation (slow, uses LLM)
        sample_step: Step for sampling questions in reply eval
    
    Returns:
        Dictionary with test results
    """
    rag = models.RAG(**config)
    rag.load_wikipedia_files(WIKI_FILES)
    
    result = {**config}
    
    # Retrieval evaluation
    retrieval = evaluate_retrieval(rag, DF)
    result["nb_chunks"] = retrieval["nb_chunks"]
    result["mrr"] = retrieval["mrr"]
    
    # Reply evaluation (optional, slow)
    if run_reply:
        df_sample = DF.iloc[::sample_step] if sample_step > 1 else DF
        reply = evaluate_reply(rag, df_sample)
        result["percent_correct"] = reply["percent_correct"]
        result["reply_similarity"] = reply["reply_similarity"]
    
    return result


def test_embeddings_and_chunk_sizes():
    """Test 1: Different embeddings with different chunk sizes."""
    print("\n" + "="*60)
    print("TEST 1: Embeddings × Chunk Sizes")
    print("="*60)
    
    embeddings = ["miniLM", "mpnet", "bge", "e5", "gte"]
    chunk_sizes = [128, 256, 512]
    
    results = []
    for embedding in embeddings:
        for chunk_size in chunk_sizes:
            print(f"\nTesting: {embedding} / chunk_size={chunk_size}")
            config = {"chunk_size": chunk_size, "embedding": embedding}
            result = run_test(config, run_reply=True)
            results.append(result)
            print(f"  → MRR: {result['mrr']:.4f}, percent_correct: {result.get('percent_correct', 'N/A')}")
    
    return pd.DataFrame(results)


def test_chunk_sizes(embedding: str):
    """Test 2: Different chunk sizes with best embedding."""
    print("\n" + "="*60)
    print(f"TEST 2: Chunk Sizes (embedding={embedding})")
    print("="*60)
    
    chunk_sizes = [128, 256, 512, 1024]
    
    results = []
    for chunk_size in chunk_sizes:
        print(f"\nTesting: chunk_size={chunk_size}")
        config = {"chunk_size": chunk_size, "embedding": embedding}
        result = run_test(config, run_reply=True)
        results.append(result)
        print(f"  → MRR: {result['mrr']:.4f}, percent_correct: {result.get('percent_correct', 'N/A')}")
    
    return pd.DataFrame(results)


def test_chunk_sizes_with_overlap(embedding: str):
    """Test 3: Chunk sizes with ~10% overlap."""
    print("\n" + "="*60)
    print(f"TEST 3: Chunk Sizes + Overlap (embedding={embedding})")
    print("="*60)
    
    configs = [
        {"chunk_size": 128, "overlap": 12},
        {"chunk_size": 256, "overlap": 25},
        {"chunk_size": 512, "overlap": 51},
        {"chunk_size": 1024, "overlap": 102},
    ]
    
    results = []
    for cfg in configs:
        print(f"\nTesting: chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']}")
        config = {**cfg, "embedding": embedding}
        result = run_test(config, run_reply=True)
        results.append(result)
        print(f"  → MRR: {result['mrr']:.4f}, percent_correct: {result.get('percent_correct', 'N/A')}")
    
    return pd.DataFrame(results)


def test_chunk_sizes_with_small2big(embedding: str):
    """Test 4: Chunk sizes with small2big."""
    print("\n" + "="*60)
    print(f"TEST 4: Chunk Sizes + Small2Big (embedding={embedding})")
    print("="*60)
    
    chunk_sizes = [128, 256, 512, 1024]
    
    results = []
    for chunk_size in chunk_sizes:
        print(f"\nTesting: chunk_size={chunk_size}, small2big=True")
        config = {"chunk_size": chunk_size, "small2big": True, "embedding": embedding}
        result = run_test(config, run_reply=True)
        results.append(result)
        print(f"  → MRR: {result['mrr']:.4f}, percent_correct: {result.get('percent_correct', 'N/A')}")
    
    return pd.DataFrame(results)


def test_chunk_sizes_with_overlap_and_metadata(embedding: str):
    """Test 5: Chunk sizes with overlap and metadata."""
    print("\n" + "="*60)
    print(f"TEST 5: Chunk Sizes + Overlap + Metadata (embedding={embedding})")
    print("="*60)
    
    configs = [
        {"chunk_size": 128, "overlap": 12},
        {"chunk_size": 256, "overlap": 25},
        {"chunk_size": 512, "overlap": 51},
        {"chunk_size": 1024, "overlap": 102},
    ]
    
    results = []
    for cfg in configs:
        print(f"\nTesting: chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']}, add_metadata=True")
        config = {**cfg, "add_metadata": True, "embedding": embedding}
        result = run_test(config, run_reply=True)
        results.append(result)
        print(f"  → MRR: {result['mrr']:.4f}, percent_correct: {result.get('percent_correct', 'N/A')}")
    
    return pd.DataFrame(results)


def select_best_embedding(df_embeddings: pd.DataFrame) -> str:
    """Select the best embedding based on combined MRR and percent_correct."""
    # Group by embedding and compute mean scores
    grouped = df_embeddings.groupby("embedding").agg({
        "mrr": "mean",
        "percent_correct": "mean"
    }).reset_index()
    
    # Combined score (weighted average)
    grouped["combined_score"] = grouped["mrr"] * 0.4 + grouped["percent_correct"] * 0.6
    
    best = grouped.loc[grouped["combined_score"].idxmax()]
    print(f"\n→ Best embedding: {best['embedding']} (MRR={best['mrr']:.4f}, percent_correct={best['percent_correct']:.4f})")
    
    return best["embedding"]


def run_all_tests():
    """Run all tests and save results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    # Test 1: Embeddings × Chunk Sizes
    df1 = test_embeddings_and_chunk_sizes()
    df1.to_csv(results_dir / f"test1_embeddings_{timestamp}.csv", index=False)
    all_results["test1_embeddings"] = df1.to_dict(orient="records")
    
    # Select best embedding
    best_embedding = select_best_embedding(df1)
    all_results["best_embedding"] = best_embedding
    
    # Test 2: Chunk Sizes
    df2 = test_chunk_sizes(best_embedding)
    df2.to_csv(results_dir / f"test2_chunk_sizes_{timestamp}.csv", index=False)
    all_results["test2_chunk_sizes"] = df2.to_dict(orient="records")
    
    # Test 3: Chunk Sizes + Overlap
    df3 = test_chunk_sizes_with_overlap(best_embedding)
    df3.to_csv(results_dir / f"test3_overlap_{timestamp}.csv", index=False)
    all_results["test3_overlap"] = df3.to_dict(orient="records")
    
    # Test 4: Small2Big
    df4 = test_chunk_sizes_with_small2big(best_embedding)
    df4.to_csv(results_dir / f"test4_small2big_{timestamp}.csv", index=False)
    all_results["test4_small2big"] = df4.to_dict(orient="records")
    
    # Test 5: Overlap + Metadata
    df5 = test_chunk_sizes_with_overlap_and_metadata(best_embedding)
    df5.to_csv(results_dir / f"test5_metadata_{timestamp}.csv", index=False)
    all_results["test5_metadata"] = df5.to_dict(orient="records")
    
    # Save all results
    with open(results_dir / f"all_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {results_dir}/")
    
    return all_results


def run_retrieval_tests_only(best_embedding: str = "miniLM"):
    """Run all tests with retrieval only (no LLM calls - faster and no rate limits)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    all_results = {"best_embedding": best_embedding}
    
    def run_configs(configs: list[dict], test_name: str):
        results = []
        for cfg in configs:
            print(f"\n  {cfg}")
            config = {**cfg, "embedding": best_embedding}
            rag = models.RAG(**config)
            rag.load_wikipedia_files(WIKI_FILES)
            retrieval = evaluate_retrieval(rag, DF)
            result = {**cfg, "nb_chunks": retrieval["nb_chunks"], "mrr": retrieval["mrr"]}
            results.append(result)
            print(f"    → MRR: {result['mrr']:.4f}, nb_chunks: {result['nb_chunks']}")
        return pd.DataFrame(results)
    
    # Test 2: Chunk Sizes
    print("\n" + "="*60)
    print(f"TEST 2: Chunk Sizes (embedding={best_embedding})")
    print("="*60)
    configs = [{"chunk_size": cs} for cs in [128, 256, 512, 1024]]
    df2 = run_configs(configs, "chunk_sizes")
    df2.to_csv(results_dir / f"test2_chunk_sizes_{timestamp}.csv", index=False)
    all_results["test2_chunk_sizes"] = df2.to_dict(orient="records")
    
    # Test 3: Chunk Sizes + Overlap
    print("\n" + "="*60)
    print(f"TEST 3: Chunk Sizes + Overlap (embedding={best_embedding})")
    print("="*60)
    configs = [
        {"chunk_size": 128, "overlap": 12},
        {"chunk_size": 256, "overlap": 25},
        {"chunk_size": 512, "overlap": 51},
        {"chunk_size": 1024, "overlap": 102},
    ]
    df3 = run_configs(configs, "overlap")
    df3.to_csv(results_dir / f"test3_overlap_{timestamp}.csv", index=False)
    all_results["test3_overlap"] = df3.to_dict(orient="records")
    
    # Test 4: Small2Big
    print("\n" + "="*60)
    print(f"TEST 4: Chunk Sizes + Small2Big (embedding={best_embedding})")
    print("="*60)
    configs = [{"chunk_size": cs, "small2big": True} for cs in [128, 256, 512, 1024]]
    df4 = run_configs(configs, "small2big")
    df4.to_csv(results_dir / f"test4_small2big_{timestamp}.csv", index=False)
    all_results["test4_small2big"] = df4.to_dict(orient="records")
    
    # Test 5: Overlap + Metadata
    print("\n" + "="*60)
    print(f"TEST 5: Chunk Sizes + Overlap + Metadata (embedding={best_embedding})")
    print("="*60)
    configs = [
        {"chunk_size": 128, "overlap": 12, "add_metadata": True},
        {"chunk_size": 256, "overlap": 25, "add_metadata": True},
        {"chunk_size": 512, "overlap": 51, "add_metadata": True},
        {"chunk_size": 1024, "overlap": 102, "add_metadata": True},
    ]
    df5 = run_configs(configs, "metadata")
    df5.to_csv(results_dir / f"test5_metadata_{timestamp}.csv", index=False)
    all_results["test5_metadata"] = df5.to_dict(orient="records")
    
    # Save all results
    with open(results_dir / f"retrieval_results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {results_dir}/")
    return all_results


def run_retrieval_only_tests():
    """Run faster tests (retrieval only, no LLM calls)."""
    print("\n" + "="*60)
    print("QUICK TEST: Retrieval Only (no LLM)")
    print("="*60)
    
    embeddings = ["miniLM", "mpnet", "bge"]
    chunk_sizes = [128, 256, 512]
    
    results = []
    for embedding in embeddings:
        for chunk_size in chunk_sizes:
            print(f"\nTesting: {embedding} / chunk_size={chunk_size}")
            config = {"chunk_size": chunk_size, "embedding": embedding}
            result = run_test(config, run_reply=False)
            results.append(result)
            print(f"  → MRR: {result['mrr']:.4f}, nb_chunks: {result['nb_chunks']}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test (retrieval only)
        df = run_retrieval_only_tests()
        print("\n" + df.to_string())
    else:
        # Full test suite
        results = run_all_tests()

