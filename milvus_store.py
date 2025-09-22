import os
from typing import List, Optional
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from pymilvus import connections, utility, Collection, MilvusException

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
        connection_args: Optional[dict] = None
    ):
        self.collection_name = collection_name
        self.embedding = OpenAIEmbeddings(model=embedding_model)

        if connection_args is None:
            self.connection_args = {
                "uri": os.getenv("MILVUS_URI", "./milvus_local.db"),
            }
            if HAS_STREAMLIT:
                milvus_uri = os.getenv("MILVUS_URI", st.secrets.get("MILVUS_URI", "./milvus_local.db"))
                self.connection_args["uri"] = milvus_uri

            milvus_token = os.getenv("MILVUS_TOKEN", None)
            if HAS_STREAMLIT and not milvus_token:
                milvus_token = st.secrets.get("MILVUS_TOKEN", None)

            if milvus_token:
                self.connection_args["token"] = milvus_token
        else:
            self.connection_args = connection_args

        self.vector_store = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize or load the Milvus vector store."""
        try:
            # Add sync configuration to avoid async issues in Streamlit
            connection_args = self.connection_args.copy()
            connection_args.update({
                "prefer_grpc": False,
                "timeout": 30
            })

            self.vector_store = Milvus(
                embedding_function=self.embedding,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=True,
                drop_old=False
            )
        except Exception as e:
            print(f"Error initializing Milvus: {e}")
            connection_args = self.connection_args.copy()
            connection_args.update({
                "prefer_grpc": False,
                "timeout": 30
            })

            self.vector_store = Milvus(
                embedding_function=self.embedding,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=True,
                drop_old=True
            )

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Add texts to the vector store."""
        if not self.vector_store:
            self._initialize_store()

        if metadatas:
            return self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        else:
            return self.vector_store.add_texts(texts=texts)

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not self.vector_store:
            self._initialize_store()

        return self.vector_store.add_documents(documents=documents)

    def similarity_search(self, query: str, k: int = 4):
        """Perform similarity search."""
        if not self.vector_store:
            self._initialize_store()

        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4):
        """Perform similarity search with scores."""
        if not self.vector_store:
            self._initialize_store()

        return self.vector_store.similarity_search_with_score(query, k=k)

    def delete(self, ids: Optional[List[str]] = None):
        """Delete documents from the collection."""
        if not self.vector_store:
            return

        if ids:
            self.vector_store.delete(ids=ids)

    def get_collection_stats(self):
        """Get collection statistics."""
        try:
            connections.connect(alias="default", **self.connection_args)
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                collection.load()
                stats = {
                    "num_entities": collection.num_entities,
                    "description": collection.description,
                    "schema": str(collection.schema)
                }
                return stats
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
        connection_args: Optional[dict] = None
    ):
        """Create a MilvusVectorStore from texts."""
        store = cls(
            collection_name=collection_name,
            embedding_model=embedding_model,
            connection_args=connection_args
        )
        store.add_texts(texts, metadatas)
        return store

    @classmethod
    def migrate_from_faiss(cls, faiss_path: str = "faiss_index"):
        """Migrate data from FAISS to Milvus."""
        from langchain_community.vectorstores import FAISS

        try:
            embedding = OpenAIEmbeddings(model="text-embedding-3-large")
            faiss_store = FAISS.load_local(
                faiss_path,
                embedding,
                allow_dangerous_deserialization=True
            )

            texts = []
            metadatas = []

            for doc_id in faiss_store.index_to_docstore_id.values():
                doc = faiss_store.docstore.search(doc_id)
                if doc:
                    texts.append(doc.page_content)
                    metadatas.append(doc.metadata if doc.metadata else {})

            milvus_store = cls.from_texts(
                texts=texts,
                metadatas=metadatas
            )

            print(f"Successfully migrated {len(texts)} documents from FAISS to Milvus")
            return milvus_store

        except Exception as e:
            print(f"Error during migration: {e}")
            return None