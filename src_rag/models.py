import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import tiktoken
import openai
import yaml
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config / Groq
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = ROOT / "config.yml"
CONF = yaml.safe_load(open(CONF_PATH, encoding="utf-8")) if CONF_PATH.exists() else {}

# ✅ Priorité à la variable d'environnement (PyCharm / Conda)
# puis fallback sur config.yml (clé "groq_key")
GROQ_API_KEY = (
        os.getenv("GROQ_API_KEY")
        or os.getenv("GROQ_KEY")
        or (CONF.get("groq_key") if isinstance(CONF, dict) else None)
)

if not GROQ_API_KEY:
    raise ValueError(
        "Clé Groq introuvable. Définis GROQ_API_KEY dans tes variables d'environnement "
        "ou ajoute 'groq_key:' dans config.yml."
    )

CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

tokenizer = tiktoken.get_encoding("cl100k_base")


# ---------------------------
# Wikipedia TXT parsing utils
# ---------------------------
def parse_wikipedia_txt(raw_text: str, filename: Optional[Path] = None) -> Tuple[str, Dict[str, str]]:
    """
    Parse les fichiers txt Wikipédia de ton format :

    Entity: ...
    Wikipedia Title: ...
    URL: ...
    Region: ...
    Period: ...
    ===========================
    <contenu...>

    Retourne:
      - content_text : le texte après la ligne de ======
      - meta : dict avec les champs extraits (entity, title, url, region, period + autres)
    """
    text = raw_text or ""
    lines = text.splitlines()

    # Trouver la 1ère ligne séparateur (>= 5 '=')
    sep_idx = None
    sep_re = re.compile(r"^\s*={5,}\s*$")
    for i, line in enumerate(lines):
        if sep_re.match(line):
            sep_idx = i
            break

    # Si pas de séparateur, on considère tout comme contenu
    if sep_idx is None:
        return text, {}

    header_lines = lines[:sep_idx]
    content_lines = lines[sep_idx + 1:]

    meta: Dict[str, str] = {}
    kv_re = re.compile(r"^\s*([^:]{1,80})\s*:\s*(.*)\s*$")

    for line in header_lines:
        line = line.strip()
        if not line:
            continue
        m = kv_re.match(line)
        if m:
            k = m.group(1).strip()
            v = m.group(2).strip()
            meta[k] = v

    # Normalisation des clés "connues"
    normalized = {}
    key_map = {
        "Entity": "entity",
        "Wikipedia Title": "title",
        "URL": "url",
        "Region": "region",
        "Period": "period",
    }
    for k, v in meta.items():
        normalized[key_map.get(k, k)] = v

    content_text = "\n".join(content_lines).strip()
    # sécurité: si contenu vide, fallback sur raw_text
    if not content_text:
        content_text = text

    return content_text, normalized


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

            p = Path(filename)

            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                raw_txt = f.read()

            # ✅ Nettoyage + meta wikipedia si présent
            cleaned_txt, meta = parse_wikipedia_txt(raw_txt, p)

            self._loaded_files.add(filename)
            self._texts.append(cleaned_txt)

            # ✅ titre prioritaire: meta["title"] si dispo
            title = meta.get("title") or _guess_title(cleaned_txt, p)

            # ✅ meta_prefix enrichi
            meta_bits = [f"[SOURCE: {p.name}]", f"[TITLE: {title}]"]
            if meta.get("entity"):
                meta_bits.append(f"[ENTITY: {meta['entity']}]")
            if meta.get("region"):
                meta_bits.append(f"[REGION: {meta['region']}]")
            if meta.get("period"):
                meta_bits.append(f"[PERIOD: {meta['period']}]")
            if meta.get("url"):
                meta_bits.append(f"[URL: {meta['url']}]")

            meta_prefix = " ".join(meta_bits) + "\n"

            # chunks du document, préfixés par les meta
            chunks_added = chunk_markdown(
                cleaned_txt,
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
        scores = (q @ self._corpus_embedding.T)[0]  # (N,)

        if not self._small2big:
            # ✅ Top-k du MEILLEUR au MOINS BON
            idxs = np.argsort(scores)[::-1][: self._top_k]
            return [self._chunks[int(i)] for i in idxs]

        # -------------------------
        # small2big: prendre plus de petits chunks, fusionner contigus
        # MAIS retourner les groupes par score décroissant
        # -------------------------
        top_small = np.argsort(scores)[::-1][: self._top_k_small]  # top scores (ranked)
        pos_sorted = sorted(int(i) for i in top_small)  # tri par position pour fusionner

        groups = []
        group = [pos_sorted[0]]
        for idx in pos_sorted[1:]:
            if idx == group[-1] + 1:
                group.append(idx)
            else:
                groups.append(group)
                group = [idx]
        groups.append(group)

        # ✅ scorer chaque groupe (max score des chunks du groupe) puis trier
        scored_groups = []
        for g in groups:
            g_score = float(max(scores[i] for i in g))
            g_text = "\n".join(self._chunks[i] for i in g)
            scored_groups.append((g_score, g_text))

        scored_groups.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_groups[: self._top_k]]

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
