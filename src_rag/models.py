import numpy as np
import re
import tiktoken
import openai
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

# ---------------------------
# Config / Groq
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
CONF = yaml.safe_load(open(ROOT / "config.yml", encoding="utf-8"))

CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONF["groq_key"],
)

tokenizer = tiktoken.get_encoding("cl100k_base")


def _guess_title(text: str, filename: Path) -> str:
    """
    Heuristique simple :
    - tente de prendre la 1ère ligne non vide (si courte)
    - sinon fallback sur le nom de fichier (stem)
    """
    for line in text.splitlines():
        line = line.strip()
        if line:
            if 5 <= len(line) <= 120:
                return line
            break
    return filename.stem.replace("_", " ").replace("-", " ")


def get_model(config: Dict[str, Any] | None):
    """
    config attendu:
    {
      "model": {
        "chunk_size": int,
        "overlap": int,
        "small2big": bool,
        "embedding": str
      }
    }
    """
    if config and "model" in config:
        return RAG(**config["model"])
    return RAG()


# ---------------------------
# RAG
# ---------------------------
class RAG:
    def __init__(
            self,
            chunk_size: int = 256,
            overlap: int = 0,
            small2big: bool = False,
            embedding: str = "miniLM",
            top_k: int = 5,
            top_k_small: int = 10,
            batch_size: int = 16,
    ):
        self._chunk_size = int(chunk_size)
        self._overlap = int(overlap)
        self._small2big = bool(small2big)
        self._embedding_name = str(embedding)

        self._top_k = int(top_k)
        self._top_k_small = int(top_k_small)
        self._batch_size = int(batch_size)

        self._embedder: Optional[SentenceTransformer] = None
        self._loaded_files = set()

        self._texts: List[str] = []
        self._chunks: List[str] = []
        self._corpus_embedding: Optional[np.ndarray] = None

        self._client = CLIENT

    # ---------------------------
    # Load + Chunk + Embed
    # ---------------------------
    def load_files(self, filenames):
        for filename in filenames:
            if filename in self._loaded_files:
                continue

            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()

            self._loaded_files.add(filename)
            self._texts.append(txt)

            title = _guess_title(txt, Path(filename))
            meta_prefix = f"[SOURCE: {Path(filename).name}] [TITLE: {title}]\n"

            # chunks du document, préfixés par les meta
            chunks_added = chunk_markdown(
                txt,
                chunk_size=self._chunk_size,
                overlap=self._overlap,
                meta_prefix=meta_prefix,
            )

            self._chunks += chunks_added

            new_embedding = self.embed_corpus(chunks_added)
            if self._corpus_embedding is None:
                self._corpus_embedding = new_embedding
            else:
                self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])

    def _compute_chunks(self, texts: List[str]) -> List[str]:
        return sum(
            (chunk_markdown(txt, chunk_size=self._chunk_size, overlap=self._overlap) for txt in texts),
            [],
        )

    def get_chunks(self) -> List[str]:
        return self._chunks

    # ---------------------------
    # Embedder
    # ---------------------------
    def get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            # Alias -> HF name
            model_name_map = {
                # Baselines EN
                "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
                "miniLM12": "sentence-transformers/all-MiniLM-L12-v2",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
                # Multilingual (FR-friendly) ✅
                "mm_mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "mm_minilm12": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "distiluse_multi": "sentence-transformers/distiluse-base-multilingual-cased-v2",
            }

            hf_name = model_name_map.get(self._embedding_name, self._embedding_name)

            self._embedder = SentenceTransformer(
                hf_name,
                device="cpu",
            )
        return self._embedder

    def embed_corpus(self, chunks: List[str]) -> np.ndarray:
        embedder = self.get_embedder()
        return embedder.encode(
            chunks,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # ✅ mieux pour dot product
        )

    def embed_questions(self, questions: List[str]) -> np.ndarray:
        embedder = self.get_embedder()
        return embedder.encode(
            questions,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # ✅ cohérence avec corpus
        )

    # ---------------------------
    # Retrieval
    # ---------------------------
    def _get_context(self, query: str) -> List[str]:
        if self._corpus_embedding is None or len(self._chunks) == 0:
            return []

        q = self.embed_questions([query])  # (1, d)
        scores = q @ self._corpus_embedding.T  # (1, N) car embeddings normalisés => cosine

        if not self._small2big:
            # Top-k chunks
            idxs = np.argsort(scores[0])[-self._top_k:]
            idxs = idxs.tolist()
            return [self._chunks[i] for i in idxs]

        # small2big: prend plus de petits chunks, fusionne ceux contigus
        idxs = np.argsort(scores[0])[-self._top_k_small:]
        idxs = sorted(idxs.tolist())

        merged = []
        group = [idxs[0]]
        for idx in idxs[1:]:
            if idx == group[-1] + 1:
                group.append(idx)
            else:
                merged.append("\n".join(self._chunks[i] for i in group))
                group = [idx]
        merged.append("\n".join(self._chunks[i] for i in group))

        return merged[: self._top_k]

    # ---------------------------
    # LLM Answer (Groq)
    # ---------------------------
    def reply(self, query: str) -> str:
        context_str = "\n\n".join(self._get_context(query))

        prompt = f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply "I cannot answer that question".
Query: {query}
Answer:"""

        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return res.choices[0].message.content


# ---------------------------
# Chunking utils
# ---------------------------
def parse_markdown_sections(md_text: str) -> List[Dict[str, str]]:
    """
    Découpe un texte markdown (ou txt) en sections basée sur headers markdown (#, ##, ...)
    Si ton corpus est .txt sans headers, ça marche quand même (tout dans une section).
    """
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []
    header_stack = []
    current_section = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)
        if match:
            if current_section["content"].strip():
                sections.append(current_section)

            level = len(match.group(1))
            title = match.group(2).strip()

            header_stack = header_stack[: level - 1]
            header_stack.append(title)

            current_section = {"headers": header_stack.copy(), "content": ""}
        else:
            current_section["content"] += line + "\n"

    if current_section["content"].strip():
        sections.append(current_section)

    # Si aucun header et sections vide, fallback
    if not sections:
        sections = [{"headers": [], "content": md_text}]
    return sections


def chunk_markdown(md_text: str, chunk_size: int = 256, overlap: int = 0, meta_prefix: str = "") -> List[str]:
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) doit être < chunk_size ({chunk_size})")

    sections = parse_markdown_sections(md_text)
    chunks: List[str] = []

    step = chunk_size - overlap

    for sec in sections:
        tokens = tokenizer.encode(sec["content"])
        for i in range(0, len(tokens), step):
            window = tokens[i: i + chunk_size]
            if not window:
                continue

            chunk_text = tokenizer.decode(window)
            chunks.append(meta_prefix + chunk_text)
            if i + chunk_size >= len(tokens):
                break

    return chunks
