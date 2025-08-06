# RAG-Based Domain-Specific Q&A System

A comprehensive Retrieval-Augmented Generation (RAG) system that provides accurate answers to user questions by leveraging custom documents as the knowledge base.

## 🚀 Features

- **Multi-format Document Support**: PDF, DOCX, TXT, PPTX, XLSX
- **Advanced RAG Pipeline**: Context-aware question answering
- **Vector Database**: Efficient similarity search with FAISS
- **User-friendly Interface**: Streamlit-based web application
- **Real-time Processing**: Instant document processing and querying
- **Source Attribution**: References to source documents and sections

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-chatbot-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main_app.py
```

## 🎯 Usage

### Quick Start
1. Launch the application: `streamlit run main_app.py`
2. Upload documents using the sidebar file uploader
3. Click "Process Documents" to create the vector store
4. Start asking questions in the chat interface

### Supported Document Types
- PDF files (.pdf)
- Word documents (.docx)
- Text files (.txt)
- PowerPoint presentations (.pptx)
- Excel spreadsheets (.xlsx)

### Sample Questions to Try
- "What is the employee's base salary?"
- "How many vacation days do employees get?"
- "What is the work from home policy?"
- "What benefits are provided?"

## 🏗️ Architecture

```
rag-chatbot-system/
├── main_app.py              # Main Streamlit application
├── document_processor.py    # Document loading and processing
├── vector_store.py         # Vector database management
├── rag_pipeline.py         # RAG pipeline implementation
├── config.py              # Configuration settings
├── requirements.txt       # Dependencies
├── sample_documents/      # Sample documents for testing
└── README.md             # This file
```

## 🔧 Configuration

Edit `config.py` to customize:
- Embedding model
- LLM model
- Chunk size and overlap
- Retrieval parameters

## 🧪 Testing

The system includes sample documents for testing:
- `sample_legal_contract.txt` - Employment agreement
- `sample_company_policy.txt` - Company policies

## 📊 Performance

- **Processing Speed**: ~100 pages/minute (depends on document complexity)
- **Query Response**: ~2-5 seconds per question
- **Memory Usage**: ~2-4GB RAM for typical usage

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU
   - Use smaller models

2. **Document Processing Fails**
   - Check file format compatibility
   - Ensure files are not corrupted

3. **Slow Response Times**
   - Reduce chunk size
   - Use smaller embedding models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support, please open an issue on GitHub or contact the development team.
