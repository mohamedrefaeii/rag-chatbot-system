import os
import streamlit as st
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Any

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_single_document(self, file_path: str) -> List[Any]:
        """Load a single document based on its file extension."""
        file_extension = file_path.split(".")[-1].lower()
        
        try:
            if file_extension == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == "docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == "txt":
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == "pptx":
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_extension in ["xlsx", "xls"]:
                loader = UnstructuredExcelLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                return []
            
            documents = loader.load()
            return documents
            
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        """Load multiple documents."""
        all_documents = []
        for file_path in file_paths:
            documents = self.load_single_document(file_path)
            all_documents.extend(documents)
        return all_documents
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def process_uploaded_files(self, uploaded_files) -> List[Any]:
        """Process uploaded files and return document chunks."""
        if not uploaded_files:
            return []
        
        # Save uploaded files temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Load and process documents
        documents = self.load_documents(file_paths)
        chunks = self.split_documents(documents)
        
        # Clean up temporary files
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return chunks
    
    def get_document_stats(self, chunks: List[Any]) -> dict:
        """Get statistics about processed documents."""
        total_chunks = len(chunks)
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "total_characters": total_chars,
            "average_chunk_size": round(avg_chunk_size, 2)
        }
