import numpy as np
import re
import tiktoken
import openai
import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CONF = yaml.safe_load(open(ROOT / "config.yml"))

CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONF["groq_key"],
)

tokenizer = tiktoken.get_encoding("cl100k_base")


def get_model(config):
    if config:
        return RAG(**config.get("model", {}))
    else:
        return RAG()


class RAG:
    def __init__(self, chunk_size: int = 256, overlap: int = 0, small2big: bool = False, embedding: str = "miniLM", add_metadata: bool = False):
        self._chunk_size = chunk_size
        if overlap > 0 and small2big:
            overlap = 0
            print("overlap is set to 0 because small2big is True")
        self._overlap = overlap
        self._small2big = small2big
        self._embedding_name = embedding
        self._embedder = None
        self._loaded_files = set()
        self._texts: list[str] = []
        self._chunks: list[str] = []
        self._corpus_embedding: np.ndarray | None = None
        self._client = CLIENT
        self._add_metadata = add_metadata

    # ---------------------------
    # Chargement & chunks
    # ---------------------------
    def load_files(self, filenames):
        texts = []
        for filename in filenames:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        self._texts += texts

        chunks_added = self._compute_chunks(texts)
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding

    def load_wikipedia_files(self, filenames):
        """Load Wikipedia format files (with Entity/Title/URL/Region/Period headers)."""
        texts = []
        for filename in filenames:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        self._texts += texts

        chunks_added = self._compute_wikipedia_chunks(texts)
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding

    def _compute_chunks(self, texts):
        return sum(
            (chunk_markdown(txt, chunk_size=self._chunk_size, overlap=self._overlap, add_metadata=self._add_metadata) for txt in texts),
            [],
        )

    def _compute_wikipedia_chunks(self, texts):
        return sum(
            (chunk_wikipedia(txt, chunk_size=self._chunk_size, overlap=self._overlap, add_metadata=self._add_metadata) for txt in texts),
            [],
        )

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    # ---------------------------
    # Embeddings
    # ---------------------------
    EMBEDDING_MODELS = {
        "miniLM": "all-MiniLM-L6-v2",        # léger, bon baseline
        "mpnet": "all-mpnet-base-v2",        # plus lourd, souvent meilleur
        "bge": "BAAI/bge-base-en-v1.5",      # version SentenceTransformers de BGE
        "e5": "intfloat/e5-base-v2",         # bon pour retrieval
        "gte": "thenlper/gte-base",          # performant et compact
    }

    def get_embedder(self):
        if self._embedder is None:
            hf_name = self.EMBEDDING_MODELS.get(self._embedding_name, self._embedding_name)

            self._embedder = SentenceTransformer(
                hf_name,
                device="cpu",  # on reste sur CPU pour éviter les galères GPU Windows
            )
        return self._embedder

    def set_embedding(self, embedding: str, recompute: bool = True):
        """
        Switch to a different embedding model.
        
        Args:
            embedding: Model name (miniLM, mpnet, bge, e5, gte) or HuggingFace model path
            recompute: If True, re-embed the corpus with the new model
        
        Available models:
            - miniLM: all-MiniLM-L6-v2 (fast, good baseline)
            - mpnet: all-mpnet-base-v2 (heavier, often better)
            - bge: BAAI/bge-base-en-v1.5 (good for retrieval)
            - e5: intfloat/e5-base-v2 (strong retrieval model)
            - gte: thenlper/gte-base (performant and compact)
        """
        if embedding == self._embedding_name:
            return  # Already using this model
        
        self._embedding_name = embedding
        self._embedder = None  # Reset cached embedder
        
        if recompute and self._chunks:
            print(f"Re-embedding {len(self._chunks)} chunks with '{embedding}'...")
            self._corpus_embedding = self.embed_corpus(self._chunks)
            print("Done.")

    def list_embeddings(self) -> dict[str, str]:
        """Return available embedding models."""
        return self.EMBEDDING_MODELS.copy()

    def embed_corpus(self, chunks):
        embedder = self.get_embedder()
        return embedder.encode(
            chunks,
            batch_size=16,  # limite l’empreinte mémoire
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def embed_questions(self, questions):
        embedder = self.get_embedder()
        return embedder.encode(
            questions,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    # ---------------------------
    # RAG : retrieval + LLM Groq
    # ---------------------------
    def reply(self, query: str) -> str:
        prompt = self._build_prompt(query)
        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return res.choices[0].message.content

    def _build_prompt(self, query: str) -> str:
        context_str = "\n".join(self._get_context(query))

        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply "I cannot answer that question".
Query: {query}
Answer:"""

    def _get_context(self, query: str) -> list[str]:
        # 1) embed la question
        query_embedding = self.embed_questions([query])  # (1, d)
        # 2) scores de similarité avec tout le corpus
        sim_scores = query_embedding @ self._corpus_embedding.T  # (1, N)

        if not self._small2big:
            # Comportement classique : top-5 chunks indépendants
            top_k = 5
            idxs = list(np.argsort(sim_scores[0]))[-top_k:]
            return [self._chunks[i] for i in idxs]

        # small2Big = True → on sélectionne les 10 chunks les plus similaires et on les fusionne avec leurs chunks adjacents s'ils sont dans le top 10.
        top_k_small = 10
        idxs = list(np.argsort(sim_scores[0]))[-top_k_small:]
        idxs = sorted(idxs)  # on remet dans l'ordre du texte
        
        # Fusion des meilleurs chunks avec chunks adjacents
        # On garde (merged_chunk, max_sim_score) pour trier ensuite par pertinence
        merged_chunks_with_scores: list[tuple[str, float]] = []
        
        # Initialisation du groupe actuel avec le premier indice
        current_group = [idxs[0]]
        for idx in idxs[1:]:
            if idx == current_group[-1] + 1:
                # même zone du texte → on agrandit le groupe
                current_group.append(idx)
            else:
                # on ferme le groupe et on crée un gros chunk
                merged_text = "\n".join(self._chunks[i] for i in current_group)
                max_score = max(sim_scores[0][i] for i in current_group)
                merged_chunks_with_scores.append((merged_text, max_score))
                current_group = [idx]

        # dernier groupe
        merged_text = "\n".join(self._chunks[i] for i in current_group)
        max_score = max(sim_scores[0][i] for i in current_group)
        merged_chunks_with_scores.append((merged_text, max_score))

        # Tri par score de similarité décroissant
        merged_chunks_with_scores.sort(key=lambda x: x[1], reverse=True)

        # on limite à 5 "gros" chunks max pour le prompt
        return [chunk for chunk, _ in merged_chunks_with_scores[:5]]


# ---------------------------
# Utils : tokenisation & chunking
# ---------------------------
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def get_wikipedia_headers(raw_text: str, filename: Optional[Path] = None) -> Tuple[str, Dict[str, str]]:
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
        "Region": "region",
        "Period": "period",
    }
    for k, v in meta.items():
        if k != "URL":
            normalized[key_map.get(k, k)] = v

    content_text = "\n".join(content_lines).strip()
    # sécurité: si contenu vide, fallback sur raw_text
    if not content_text:
        content_text = text

    return content_text, normalized

def parse_markdown_sections(md_text: str) -> list[dict[str, str]]:
    """
    Parses markdown into a list of {'headers': [...], 'content': ...}
    Preserves full header hierarchy (e.g. ["Section", "Sub", "SubSub", ...])

    Example:
        Input:  "# Plot\\nThe story...\\n## Act 1\\nScene intro..."
        Output: [
            {"headers": ["Plot"], "content": "The story...\\n"},
            {"headers": ["Plot", "Act 1"], "content": "Scene intro...\\n"}
        ]
    """
    # Regex to match markdown headers: captures the # symbols (group 1) and header text (group 2)
    # Examples: "# Title" → group(1)="#", group(2)="Title"
    #           "### Subsection" → group(1)="###", group(2)="Subsection"
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []  # Final list of parsed sections

    # Stack to track header hierarchy (e.g., ["Plot", "Act 1"] for nested headers)
    header_stack = []

    # Current section being built (accumulates content until next header)
    current_section = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)

        if match:
            # --- This line is a header ---

            # Determine header level: # = 1, ## = 2, ### = 3, etc.
            level = len(match.group(1))
            title = match.group(2).strip()

            # Save the previous section if it has content (avoid empty sections)
            if current_section["content"]:
                sections.append(current_section)

            # Adjust the header stack to reflect the new hierarchy level
            # e.g., if we were at ["A", "B", "C"] (level 3) and encounter a level 2 header,
            # we slice to ["A"] then append the new title → ["A", "NewTitle"]
            header_stack = header_stack[: level - 1]
            header_stack.append(title)

            # Start a new section with the current header hierarchy
            current_section = {
                "headers": header_stack.copy(),  # Copy to avoid reference issues
                "content": "",
            }
        else:
            # --- This line is regular content → append to current section ---
            current_section["content"] += line + "\n"

    # Don't forget the last section (no header follows it to trigger save)
    if current_section["content"]:
        sections.append(current_section)

    return sections


def chunk_markdown(md_text: str, chunk_size: int = 128, overlap: int = 0, add_metadata: bool = False) -> list[str]:
    """
    Découpe le texte markdown en chunks avec overlap optionnel.
    """
    if overlap >= chunk_size:
        raise ValueError(f"Overlap ({overlap}) doit être inférieur à chunk_size ({chunk_size})")

    parsed_sections = parse_markdown_sections(md_text)
    chunks: list[str] = []

    for section in parsed_sections:
        if add_metadata:
            header_str = " > ".join(section['headers'])
            tokens = tokenizer.encode(f"[{header_str}]\n{section['content']}")
        else:
            tokens = tokenizer.encode(section["content"])
        step = chunk_size - overlap

        token_chunks = []
        # Chunk tokenized section
        for i in range(0, len(tokens), step):
            chunk = tokens[i: i + chunk_size]
            if chunk:
                token_chunks.append(chunk)
            if i + chunk_size >= len(tokens):
                break

        for token_chunk in token_chunks:
            chunk_text = tokenizer.decode(token_chunk)
            chunks.append(chunk_text)

    return chunks


def chunk_wikipedia(raw_text: str, chunk_size: int = 128, overlap: int = 0, add_metadata: bool = False) -> list[str]:
    """
    Découpe un fichier Wikipedia (format Entity/Title/URL/Region/Period + contenu) en chunks.
    
    Utilise get_wikipedia_headers pour extraire les métadonnées et le contenu,
    puis découpe le contenu en chunks avec overlap optionnel.
    """
    if overlap >= chunk_size:
        raise ValueError(f"Overlap ({overlap}) doit être inférieur à chunk_size ({chunk_size})")

    content_text, meta = get_wikipedia_headers(raw_text)
    
    # Construire le préfixe de métadonnées si demandé
    metadata_prefix = ""
    if add_metadata and meta:
        meta_parts = []
        if "entity" in meta:
            meta_parts.append(meta["entity"])
        if "region" in meta:
            meta_parts.append(meta["region"])
        if "period" in meta:
            meta_parts.append(meta["period"])
        if meta_parts:
            metadata_prefix = "[" + " | ".join(meta_parts) + "]\n"
    
    # Tokenize le contenu
    tokens = tokenizer.encode(content_text)
    
    if not tokens:
        return []
    
    step = chunk_size - overlap
    chunks: list[str] = []
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i: i + chunk_size]
        if chunk_tokens:
            chunk_text = tokenizer.decode(chunk_tokens)
            if add_metadata and metadata_prefix:
                chunk_text = metadata_prefix + chunk_text
            chunks.append(chunk_text)
        if i + chunk_size >= len(tokens):
            break
    
    return chunks
