"""
Vector Store abstraction for RAG system.
Supports both in-memory (NumPy) and ChromaDB backends.
"""
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(self, chunks: List[str], embeddings: np.ndarray, metadatas: Optional[List[dict]] = None):
        """Add chunks with their embeddings to the store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """Search for similar chunks. Returns list of (chunk, score, metadata)."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all stored data."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return number of stored chunks."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using NumPy arrays."""
    
    def __init__(self):
        self._chunks: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._metadatas: List[dict] = []
    
    def add(self, chunks: List[str], embeddings: np.ndarray, metadatas: Optional[List[dict]] = None):
        self._chunks.extend(chunks)
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        
        if metadatas:
            self._metadatas.extend(metadatas)
        else:
            self._metadatas.extend([{}] * len(chunks))
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, self._embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                self._chunks[idx],
                float(similarities[idx]),
                self._metadatas[idx] if idx < len(self._metadatas) else {}
            ))
        return results
    
    def clear(self):
        self._chunks = []
        self._embeddings = None
        self._metadatas = []
    
    def count(self) -> int:
        return len(self._chunks)
    
    def get_all_chunks(self) -> List[str]:
        return self._chunks.copy()


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store for persistent storage."""
    
    def __init__(self, collection_name: str = "rag_historian", host: str = None, port: int = None):
        import chromadb
        
        host = host or os.environ.get("CHROMADB_HOST", "localhost")
        port = port or int(os.environ.get("CHROMADB_PORT", "8000"))
        
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._chunks: List[str] = []  # Local cache for get_all_chunks
    
    def add(self, chunks: List[str], embeddings: np.ndarray, metadatas: Optional[List[dict]] = None):
        # Generate unique IDs
        start_id = self._collection.count()
        ids = [f"chunk_{start_id + i}" for i in range(len(chunks))]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{}] * len(chunks)
        
        # Convert numpy array to list of lists
        embeddings_list = embeddings.tolist()
        
        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        self._chunks.extend(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        # Ensure query is 1D list
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        if results["documents"] and results["documents"][0]:
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            ):
                # ChromaDB returns distances, convert to similarity
                similarity = 1 - dist  # For cosine distance
                output.append((doc, similarity, meta or {}))
        
        return output
    
    def clear(self):
        # Delete and recreate collection
        collection_name = self._collection.name
        self._client.delete_collection(collection_name)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._chunks = []
    
    def count(self) -> int:
        return self._collection.count()
    
    def get_all_chunks(self) -> List[str]:
        return self._chunks.copy()


def get_vector_store(use_chroma: bool = None) -> VectorStore:
    """
    Factory function to get the appropriate vector store.
    
    Args:
        use_chroma: Force ChromaDB usage. If None, auto-detect from environment.
    
    Returns:
        VectorStore instance (ChromaDB if available, otherwise in-memory)
    """
    if use_chroma is None:
        use_chroma = os.environ.get("CHROMADB_HOST") is not None
    
    if use_chroma:
        try:
            return ChromaVectorStore()
        except Exception as e:
            print(f"Warning: Could not connect to ChromaDB ({e}), using in-memory store")
            return InMemoryVectorStore()
    
    return InMemoryVectorStore()

