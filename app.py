import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile

# Initialize embeddings model
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents from uploaded files
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            loader = PyPDFLoader(uploaded_file)
            docs = loader.load()
        elif file_extension == "docx":
            loader = UnstructuredWordDocumentLoader(uploaded_file)
            docs = loader.load()
        elif file_extension == "txt":
            loader = TextLoader(uploaded_file)
            docs = loader.load()
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue
        documents.extend(docs)
    return documents

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Build or load FAISS vector store
@st.cache_resource(show_spinner=False)
def get_vectorstore(documents, embeddings):
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embeddings)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("faiss_index")
    return vectorstore

# Initialize LLM pipeline
@st.cache_resource
def get_llm():
    pipe = pipeline(
        "text-generation",
        model="gpt2",
        tokenizer="gpt2",
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
    )
    return HuggingFacePipeline(pipeline=pipe)

def main():
    st.title("RAG-Based Domain-Specific Q&A System")

    uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Loading and processing documents..."):
            documents = load_documents(uploaded_files)
            chunks = split_documents(documents)
            embeddings = get_embedding_model()
            vectorstore = get_vectorstore(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            llm = get_llm()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        query = st.text_input("Enter your question:")

        if query:
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(query)
                st.markdown("### Answer:")
                st.write(answer)
    else:
        st.info("Please upload documents to start.")

if __name__ == "__main__":
    main()
