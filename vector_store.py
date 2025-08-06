import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Any

class VectorStoreManager:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Any]) -> FAISS:
        """Create a new FAISS vector store from documents."""
        if not documents:
            raise ValueError("No documents provided")
        
        with st.spinner("Creating vector store..."):
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return self.vector_store
    
    def save_vector_store(self, path: str = "faiss_index"):
        """Save the vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(path)
            st.success(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str = "faiss_index") -> FAISS:
        """Load the vector store from disk."""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings)
            st.success("Vector store loaded successfully")
            return self.vector_store
        else:
            st.warning("No existing vector store found")
            return None
    
    def add_documents(self, documents: List[Any]):
        """Add new documents to the existing vector store."""
        if not self.vector_store:
            self.create_vector_store(documents)
        else:
            self.vector_store.add_documents(documents)
    
    def get_retriever(self, top_k: int = 3):
        """Get a retriever for similarity search."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
    
    def similarity_search(self, query: str, top_k: int = 3) -> List[Any]:
        """Perform similarity search."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=top_k)
    
    def get_vector_store_stats(self) -> dict:
        """Get statistics about the vector store."""
        if not self.vector_store:
            return {"status": "Not initialized"}
        
        # This is a simplified version - in practice, you might want to store more metadata
        return {
            "status": "Active",
            "embedding_model": self.embeddings.model_name,
            "index_type": "FAISS"
        }
