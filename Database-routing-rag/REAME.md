# 📠 Local RAG Agent with Database Routing

A powerful Retrieval-Augmented Generation (RAG) system that intelligently routes queries to specialized databases and provides comprehensive answers using local AI models.

## 🌟 Features

- **Intelligent Query Routing**: Automatically routes questions to the most relevant database (Products, Support, Finance)
- **Multi-Database Support**: Separate vector databases for different domains
- **Local AI Models**: Uses Ollama for both LLM and embedding models
- **Vector Similarity Search**: Qdrant vector database for fast similarity matching
- **Web Search Fallback**: DuckDuckGo search integration when no relevant documents are found
- **PDF Document Processing**: Upload and process PDF documents into searchable knowledge bases
- **Streamlit Web Interface**: User-friendly web interface for document upload and querying
- **Dual Routing Strategy**: Vector similarity + LLM-based routing for optimal accuracy

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │ -> │  Query Router    │ -> │   Database      │
└─────────────────┘    └──────────────────┘    │   Selection     │
                                │               └─────────────────┘
                                │                        │
                                v                        v
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Vector Similarity│    │ Products DB     │
                       │    Routing       │    │ Support DB      │
                       └─────────────────┘    │ Finance DB      │
                                │               └─────────────────┘
                                │                        │
                                v                        v
                       ┌─────────────────┐    ┌─────────────────┐
                       │  LLM Routing    │    │  Answer         │
                       │   (Fallback)    │    │  Generation     │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                v                        v
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Web Search     │    │  Final Answer   │
                       │   (Fallback)    │    │   to User       │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker (for Qdrant)
- Ollama (for local AI models)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Amityadav9/RAG-Collection.git
   cd database-routing-rag
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Qdrant using Docker:**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Install and start Ollama:**
   ```bash
   # Install from https://ollama.ai
   # Then pull the required models:
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text:latest
   ```

5. **Run the application:**
   ```bash
   streamlit run main.py
   ```

## 📋 Usage

### 1. Initialize Models
- Open the application in your browser (usually http://localhost:8501)
- Configure Ollama and Qdrant settings in the sidebar
- Click "Initialize Models" to connect to your local services

### 2. Upload Documents
- Use the document upload tabs to add PDF files to each database:
  - **Products**: Product manuals, specifications, feature descriptions
  - **Support**: FAQ documents, troubleshooting guides, customer service info
  - **Finance**: Financial reports, pricing information, cost analysis

### 3. Ask Questions
- Enter your question in the query box
- The system will automatically route your question to the most relevant database
- Get comprehensive answers based on your uploaded documents
- If no relevant documents are found, the system will search the web for answers

## 🔧 Configuration

### Ollama Models
- **LLM Model**: `llama3.2:3b` (configurable)
- **Embedding Model**: `nomic-embed-text:latest` (configurable)
- **Base URL**: `http://localhost:11434` (default)

### Qdrant Configuration
- **URL**: `http://localhost:6333` (default)
- **API Key**: Optional for local installations

### Database Collections
- **Products Collection**: `products_collection`
- **Support Collection**: `support_collection`
- **Finance Collection**: `finance_collection`

## 🎯 Query Routing Logic

The system uses a two-tier routing approach:

1. **Vector Similarity Routing** (Primary):
   - Searches all databases for similar content
   - Uses confidence scores to determine the best match
   - Threshold: 0.5 (configurable)

2. **LLM-based Routing** (Fallback):
   - Uses the agno framework with Ollama
   - Analyzes question content and context
   - Routes based on semantic understanding

3. **Web Search Fallback**:
   - Activates when no relevant documents are found
   - Uses DuckDuckGo search for current information
   - Synthesizes answers using the local LLM

## 📁 Project Structure

```
database-routing-rag/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── db_storage/            # Local database storage (created automatically)
```

## 🛠️ Dependencies

- **streamlit**: Web interface framework
- **langchain-community**: LangChain community integrations
- **langchain-core**: Core LangChain functionality
- **langchain-ollama**: Ollama integration for LangChain
- **ollama**: Python client for Ollama
- **qdrant-client**: Qdrant vector database client
- **agno**: Agent framework for LLM routing
- **pypdf**: PDF document processing
- **sentence-transformers**: Text embedding models
- **duckduckgo-search**: Web search functionality

## 🔍 How It Works

1. **Document Processing**: PDFs are loaded, split into chunks, and embedded using Ollama's embedding model
2. **Vector Storage**: Document embeddings are stored in Qdrant collections
3. **Query Processing**: User queries are embedded and compared against stored documents
4. **Smart Routing**: The system determines the best database based on similarity scores
5. **Answer Generation**: Retrieved documents are used to generate contextual answers
6. **Fallback Mechanisms**: Web search provides answers when local documents are insufficient

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local AI model hosting
- [Qdrant](https://qdrant.tech/) for vector database functionality
- [LangChain](https://langchain.com/) for RAG pipeline components
- [Streamlit](https://streamlit.io/) for the web interface
- [agno](https://github.com/agno-ai/agno) for the agent framework

## 📧 Support

If you encounter any issues or have questions, please open an issue on GitHub or contact the maintainers.

---

**Happy Querying!** 🎉
