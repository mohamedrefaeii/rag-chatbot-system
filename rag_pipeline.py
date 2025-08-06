import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Any, Dict

class RAGPipeline:
    def __init__(self, retriever, llm_model_name: str = "microsoft/DialoGPT-medium"):
        self.retriever = retriever
        self.llm = None
        self.qa_chain = None
        self.llm_model_name = llm_model_name
        
    def initialize_llm(self):
        """Initialize the language model."""
        try:
            # Try to use a better model if available
            if torch.cuda.is_available():
                device = 0
            else:
                device = -1
            
            pipe = pipeline(
                "text-generation",
                model=self.llm_model_name,
                tokenizer=self.llm_model_name,
                max_length=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                device=device,
                pad_token_id=50256  # For DialoGPT
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            return True
            
        except Exception as e:
            st.error(f"Error loading LLM: {str(e)}")
            # Fallback to a simpler model
            try:
                pipe = pipeline(
                    "text-generation",
                    model="gpt2",
                    max_length=256,
                    temperature=0.7,
                    device=-1
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
                return True
            except:
                st.error("Failed to load any LLM")
                return False
    
    def create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        if not self.llm or not self.retriever:
            raise ValueError("LLM and retriever must be initialized")
        
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite the source document and page number if available.
        
        Context:
        {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return self.qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        
        try:
            with st.spinner("Generating answer..."):
                result = self.qa_chain({"query": question})
                
                # Format the response
                response = {
                    "answer": result["result"],
                    "source_documents": []
                }
                
                # Extract source information
                for doc in result.get("source_documents", []):
                    source_info = {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    response["source_documents"].append(source_info)
                
                return response
                
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "source_documents": []
            }
    
    def get_confidence_score(self, question: str, retrieved_docs: list) -> float:
        """Calculate a simple confidence score based on similarity."""
        if not retrieved_docs:
            return 0.0
        
        # This is a simplified confidence score
        # In practice, you might want to use more sophisticated methods
        return min(1.0, len(retrieved_docs) / 3.0)
