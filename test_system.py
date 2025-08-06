"""
Test script to verify the RAG system components are working correctly.
"""

import os
import sys
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline

def test_document_processor():
    """Test document processing functionality."""
    print("Testing Document Processor...")
    
    processor = DocumentProcessor()
    
    # Create a simple test document
    test_content = "This is a test document for the RAG system. It contains information about testing and validation."
    
    # Test document splitting
    from langchain.schema import Document
    test_doc = Document(page_content=test_content, metadata={"source": "test"})
    chunks = processor.split_documents([test_doc])
    
    print(f"✓ Document processor created {len(chunks)} chunks")
    return True

def test_vector_store():
    """Test vector store functionality."""
    print("Testing Vector Store...")
    
    manager = VectorStoreManager()
    
    # Create test documents
    from langchain.schema import Document
    test_docs = [
        Document(page_content="Python is a programming language", metadata={"source": "test1"}),
        Document(page_content="Machine learning is a subset of AI", metadata={"source": "test2"})
    ]
    
    # Create vector store
    vector_store = manager.create_vector_store(test_docs)
    print("✓ Vector store created successfully")
    
    # Test similarity search
    results = manager.similarity_search("programming", top_k=1)
    print(f"✓ Similarity search returned {len(results)} results")
    
    return True

def test_rag_pipeline():
    """Test RAG pipeline initialization."""
    print("Testing RAG Pipeline...")
    
    # Create mock retriever
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    test_docs = [
        ("Python is great", {"source": "test1"}),
        ("AI is the future", {"source": "test2"})
    ]
    
    # Create simple vector store
    from langchain.schema import Document
    docs = [Document(page_content=text, metadata=meta) for text, meta in test_docs]
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    
    # Test RAG pipeline
    pipeline = RAGPipeline(retriever)
    print("✓ RAG pipeline initialized successfully")
    
    return True

def main():
    """Run all tests."""
    print("🧪 Running RAG System Tests...\n")
    
    tests = [
        test_document_processor,
        test_vector_store,
        test_rag_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                print("❌ FAILED\n")
        except Exception as e:
            print(f"❌ FAILED: {e}\n")
    
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The RAG system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
