import os
import json
import pickle
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import numpy as np
from datetime import datetime

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

class SimpleVectorStore:
    """A simple file-based vector store that avoids gRPC/async issues."""

    def __init__(
        self,
        store_path: str = "./simple_vector_store",
        embedding_model: str = "text-embedding-3-large"
    ):
        self.store_path = store_path
        self.embedding = OpenAIEmbeddings(model=embedding_model)
        self.vectors_file = os.path.join(store_path, "vectors.pkl")
        self.metadata_file = os.path.join(store_path, "metadata.json")

        # Create directory if it doesn't exist
        os.makedirs(store_path, exist_ok=True)

        # Load existing data
        self.vectors = self._load_vectors()
        self.metadata = self._load_metadata()

    def _load_vectors(self) -> List[List[float]]:
        """Load vectors from file."""
        if os.path.exists(self.vectors_file):
            try:
                with open(self.vectors_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading vectors: {e}")
        return []

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
        return []

    def _save_vectors(self):
        """Save vectors to file."""
        try:
            with open(self.vectors_file, 'wb') as f:
                pickle.dump(self.vectors, f)
        except Exception as e:
            print(f"Error saving vectors: {e}")

    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        """Add texts to the vector store."""
        if not texts:
            return []

        # Generate embeddings
        embeddings = self.embedding.embed_documents(texts)

        # Add to storage
        ids = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Create unique ID
            doc_id = f"doc_{len(self.vectors)}_{datetime.now().timestamp()}"
            ids.append(doc_id)

            # Add vector
            self.vectors.append(embedding)

            # Add metadata
            metadata = {
                "id": doc_id,
                "text": text,
                "timestamp": datetime.now().isoformat()
            }

            # Add user metadata if provided
            if metadatas and i < len(metadatas):
                metadata.update(metadatas[i])

            self.metadata.append(metadata)

        # Save to files
        self._save_vectors()
        self._save_metadata()

        print(f"Added {len(texts)} documents to vector store")
        return ids

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search."""
        if not self.vectors:
            return []

        # Generate query embedding
        query_embedding = self.embedding.embed_query(query)

        # Calculate similarities
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_embedding, vector)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top k results
        results = []
        for i, (idx, score) in enumerate(similarities[:k]):
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                text = meta.pop("text", "")
                meta.pop("id", None)  # Remove internal id
                meta.pop("timestamp", None)  # Remove timestamp unless needed

                results.append(Document(page_content=text, metadata=meta))

        return results

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores."""
        if not self.vectors:
            return []

        # Generate query embedding
        query_embedding = self.embedding.embed_query(query)

        # Calculate similarities
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_embedding, vector)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top k results with scores
        results = []
        for i, (idx, score) in enumerate(similarities[:k]):
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                text = meta.pop("text", "")
                meta.pop("id", None)
                meta.pop("timestamp", None)

                doc = Document(page_content=text, metadata=meta)
                results.append((doc, score))

        return results

    def delete(self, ids: Optional[List[str]] = None):
        """Delete documents by IDs."""
        if not ids:
            return

        # Find indices to remove
        indices_to_remove = []
        for doc_id in ids:
            for i, meta in enumerate(self.metadata):
                if meta.get("id") == doc_id:
                    indices_to_remove.append(i)
                    break

        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            if idx < len(self.vectors):
                del self.vectors[idx]
            if idx < len(self.metadata):
                del self.metadata[idx]

        # Save changes
        self._save_vectors()
        self._save_metadata()

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "document_count": len(self.metadata),
            "vector_count": len(self.vectors),
            "store_path": self.store_path,
            "status": "ready"
        }

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding_model: str = "text-embedding-3-large",
        metadatas: Optional[List[dict]] = None,
        store_path: str = "./simple_vector_store"
    ):
        """Create a SimpleVectorStore from texts."""
        store = cls(store_path=store_path, embedding_model=embedding_model)
        store.add_texts(texts, metadatas)
        return store