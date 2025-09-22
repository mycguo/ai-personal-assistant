import os
from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from pymilvus import MilvusClient, DataType
import numpy as np

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

class MilvusVectorStore:
    def __init__(
        self,
        collection_name: str = "personal_assistant",
        embedding_model: str = "text-embedding-3-large",
        uri: Optional[str] = None,
        token: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.embedding = OpenAIEmbeddings(model=embedding_model)

        if uri is None:
            uri = os.getenv("MILVUS_URI", "./milvus_local.db")
            if HAS_STREAMLIT:
                uri = os.getenv("MILVUS_URI", st.secrets.get("MILVUS_URI", "./milvus_local.db"))

        if token is None:
            token = os.getenv("MILVUS_TOKEN", None)
            if HAS_STREAMLIT and not token:
                token = st.secrets.get("MILVUS_TOKEN", None)

        # Initialize Milvus client with sync operations only
        self.client = MilvusClient(uri=uri, token=token)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the collection exists with proper schema."""
        if not self.client.has_collection(collection_name=self.collection_name):
            # Create collection with schema
            schema = self.client.create_schema(
                auto_id=True,
                enable_dynamic_field=True
            )

            # Add fields
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=3072)  # text-embedding-3-large dimension
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

            # Create index
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="FLAT",
                metric_type="COSINE"
            )

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            print(f"Created collection: {self.collection_name}")

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        """Add texts to the vector store."""
        if not texts:
            return []

        # Generate embeddings
        embeddings = self.embedding.embed_documents(texts)

        # Prepare data
        data = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            item = {
                "vector": embedding,
                "text": text
            }

            # Add metadata if provided
            if metadatas and i < len(metadatas):
                for key, value in metadatas[i].items():
                    # Convert non-string values to strings for VARCHAR compatibility
                    item[key] = str(value) if not isinstance(value, str) else value

            data.append(item)

        # Insert data
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

        return result.get("ids", [])

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search."""
        # Generate query embedding
        query_embedding = self.embedding.embed_query(query)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=k,
            output_fields=["text", "*"]  # Return text and all metadata fields
        )

        # Convert results to Documents
        documents = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                text = entity.get("text", "")

                # Extract metadata (exclude system fields)
                metadata = {}
                for key, value in entity.items():
                    if key not in ["id", "vector", "text"]:
                        metadata[key] = value

                documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores."""
        # Generate query embedding
        query_embedding = self.embedding.embed_query(query)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=k,
            output_fields=["text", "*"]
        )

        # Convert results to Documents with scores
        documents_with_scores = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                score = hit.get("distance", 0.0)
                text = entity.get("text", "")

                # Extract metadata
                metadata = {}
                for key, value in entity.items():
                    if key not in ["id", "vector", "text"]:
                        metadata[key] = value

                doc = Document(page_content=text, metadata=metadata)
                documents_with_scores.append((doc, score))

        return documents_with_scores

    def delete(self, ids: Optional[List[str]] = None):
        """Delete documents from the collection."""
        if ids:
            self.client.delete(
                collection_name=self.collection_name,
                ids=ids
            )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            if self.client.has_collection(collection_name=self.collection_name):
                stats = self.client.get_collection_stats(collection_name=self.collection_name)
                return {
                    "row_count": stats.get("row_count", 0),
                    "collection_name": self.collection_name,
                    "status": "ready"
                }
            else:
                return {"status": "Collection does not exist"}
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding_model: str = "text-embedding-3-large",
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "personal_assistant",
        uri: Optional[str] = None,
        token: Optional[str] = None
    ):
        """Create a MilvusVectorStore from texts."""
        store = cls(
            collection_name=collection_name,
            embedding_model=embedding_model,
            uri=uri,
            token=token
        )
        store.add_texts(texts, metadatas)
        return store