import os
from dotenv import load_dotenv

load_dotenv()

# Configuration settings
class Config:
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "microsoft/DialoGPT-medium"  # Better than GPT-2 for Q&A
    
    # Vector store settings
    VECTOR_STORE_PATH = "faiss_index"
    
    # Text processing settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Retrieval settings
    TOP_K_RETRIEVAL = 3
    
    # OpenAI settings (if using OpenAI API)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Hugging Face settings
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
