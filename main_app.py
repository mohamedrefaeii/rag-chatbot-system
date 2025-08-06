pip install torch
import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline
from config import Config

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stats-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class RAGChatbotApp:
    def __init__(self):
        self.config = Config()
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        self.vector_manager = VectorStoreManager(self.config.EMBEDDING_MODEL)
        self.rag_pipeline = None
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
        if 'documents_processed' not in st.session_state:
            st.session_state.documents_processed = False
    
    def sidebar_content(self):
        """Create sidebar content."""
        st.sidebar.title("📁 Document Management")
        
        # File upload
        uploaded_files = st.sidebar.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt', 'pptx', 'xlsx'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.sidebar.button("Process Documents", type="primary"):
                self.process_documents(uploaded_files)
        
        # Load existing vector store
        st.sidebar.divider()
        if st.sidebar.button("Load Existing Vector Store"):
            self.load_existing_vector_store()
        
        # Clear vector store
        st.sidebar.divider()
        if st.sidebar.button("Clear Vector Store", type="secondary"):
            self.clear_vector_store()
        
        # Settings
        st.sidebar.divider()
        st.sidebar.subheader("⚙️ Settings")
        
        top_k = st.sidebar.slider(
            "Number of retrieved documents",
            min_value=1,
            max_value=10,
            value=self.config.TOP_K_RETRIEVAL,
            key="top_k"
        )
        
        return uploaded_files
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents."""
        try:
            with st.spinner("Processing documents..."):
                # Process documents
                chunks = self.doc_processor.process_uploaded_files(uploaded_files)
                
                if not chunks:
                    st.error("No documents were successfully processed.")
                    return
                
                # Create vector store
                vector_store = self.vector_manager.create_vector_store(chunks)
                
                # Save vector store
                self.vector_manager.save_vector_store()
                
                # Initialize RAG pipeline
                retriever = self.vector_manager.get_retriever(st.session_state.get('top_k', self.config.TOP_K_RETRIEVAL))
                self.rag_pipeline = RAGPipeline(retriever)
                
                if self.rag_pipeline.initialize_llm():
                    self.rag_pipeline.create_qa_chain()
                    
                    # Update session state
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_pipeline = self.rag_pipeline
                    st.session_state.documents_processed = True
                    
                    # Show stats
                    stats = self.doc_processor.get_document_stats(chunks)
                    st.success("Documents processed successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", stats["total_chunks"])
                    with col2:
                        st.metric("Total Characters", stats["total_characters"])
                    with col3:
                        st.metric("Avg Chunk Size", f"{stats['average_chunk_size']} chars")
                        
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
    
    def load_existing_vector_store(self):
        """Load existing vector store."""
        vector_store = self.vector_manager.load_vector_store()
        if vector_store:
            st.session_state.vector_store = vector_store
            retriever = self.vector_manager.get_retriever()
            self.rag_pipeline = RAGPipeline(retriever)
            
            if self.rag_pipeline.initialize_llm():
                self.rag_pipeline.create_qa_chain()
                st.session_state.rag_pipeline = self.rag_pipeline
                st.session_state.documents_processed = True
                st.success("Vector store loaded successfully!")
    
    def clear_vector_store(self):
        """Clear the vector store."""
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        
        st.session_state.vector_store = None
        st.session_state.rag_pipeline = None
        st.session_state.documents_processed = False
        st.success("Vector store cleared!")
    
    def chat_interface(self):
        """Create the chat interface."""
        st.markdown('<h1 class="main-header">🤖 RAG Q&A System</h1>', unsafe_allow_html=True)
        
        if not st.session_state.get('documents_processed', False):
            st.info("Please upload and process documents to start chatting.")
            return
        
        # Chat interface
        st.subheader("💬 Ask Questions")
        
        # Display chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Chat input
        question = st.text_input(
            "Enter your question:",
            placeholder="Ask me anything about the uploaded documents...",
            key="question_input"
        )
        
        if question:
            if st.button("Send", type="primary"):
                # Get response
                response = st.session_state.rag_pipeline.query(question)
                
                # Add to chat history
                st.session_state.messages.append({
                    "question": question,
                    "answer": response["answer"],
                    "sources": response["source_documents"]
                })
        
        # Display chat history
        if st.session_state.messages:
            st.subheader("📋 Chat History")
            for i, msg in enumerate(reversed(st.session_state.messages)):
                with st.expander(f"Q{i+1}: {msg['question'][:50]}..."):
                    st.markdown("**Answer:**")
                    st.write(msg["answer"])
                    
                    if msg["sources"]:
                        st.markdown("**Sources:**")
                        for j, source in enumerate(msg["sources"]):
                            st.caption(f"Source {j+1}: {source['metadata'].get('source', 'Unknown')}")
    
    def main(self):
        """Main application."""
        self.initialize_session_state()
        
        # Sidebar
        self.sidebar_content()
        
        # Main content
        self.chat_interface()

if __name__ == "__main__":
    app = RAGChatbotApp()
    app.main()
