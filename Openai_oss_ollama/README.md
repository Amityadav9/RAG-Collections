# 📚 RAG Chatbot with OpenAI OSS - LangChain Edition

> **Revolutionary**: Chat with your documents using OpenAI's **first open-source model in ~5 years** - completely local and private!

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 What This Does

Transform any PDF or text document into an interactive AI assistant using **OpenAI's gpt-oss:20b** model running locally via Ollama. Upload your documents and ask questions about them in natural language!

### 🔥 Key Features

- 📄 **Multi-format Support**: PDF and TXT document processing
- 🤖 **OpenAI OSS Model**: Uses gpt-oss:20b (first open-source model since GPT-2)
- 🔒 **100% Private**: All processing happens locally on your hardware
- 💬 **Natural Conversations**: Maintains context across multiple questions
- ⚡ **Real-time Responses**: Streaming responses with typing animation
- 🎨 **Beautiful UI**: Clean Gradio interface with custom styling
- 🌐 **Remote GPU Support**: Run Ollama on powerful remote machines

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed
- At least 16GB RAM (recommended for gpt-oss:20b)

### Installation

1. **Clone and setup**:
```bash
git clone <your-repo>
cd openai_oss
pip install -r requirements.txt
```

2. **Install and configure Ollama**:
```bash
# Install Ollama from https://ollama.ai/
# Pull OpenAI's open source model
ollama pull gpt-oss:20b
```

3. **Setup environment**:
```bash
# Create .env file
echo "OLLAMA_HOST=http://localhost:11434" > .env
# For remote GPU: echo "OLLAMA_HOST=http://YOUR_GPU_IP:11434" > .env
```

4. **Run the application**:
```bash
python rag_langchain.py
```

---

## 🔧 Configuration

### 🖥️ **Local Setup (Single Machine)**
```bash
ollama serve
# App automatically connects to http://localhost:11434
```

### 🌐 **Remote Setup (Powerful GPU Machine)**

**On your GPU machine (Ubuntu example):**

1. **Create systemd service for permanent setup**:
```bash
sudo nano /etc/systemd/system/ollama.service
```

2. **Add this configuration**:
```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=your_username
Group=your_username
Restart=always
RestartSec=3
Environment="PATH=/home/your_username/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"
Environment="OLLAMA_HOST=0.0.0.0:11434"

[Install]
WantedBy=default.target
```

3. **Enable and start service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama
```

**On your client machine:**
```bash
# Update .env file
echo "OLLAMA_HOST=http://YOUR_GPU_IP:11434" > .env
```

---

## 🏗️ Architecture

### RAG Pipeline
```
📄 Document Upload
    ↓
🔪 Text Chunking (RecursiveCharacterTextSplitter)
    ↓
🧮 Vector Embeddings (sentence-transformers/all-MiniLM-L6-v2)
    ↓
🗄️ Vector Storage (FAISS)
    ↓
🔍 Similarity Search (Retrieval)
    ↓
🤖 OpenAI OSS Generation (gpt-oss:20b via Ollama)
    ↓
💬 Conversational Response
```

### Tech Stack
- **Frontend**: Gradio (Interactive web interface)
- **RAG Framework**: LangChain (Retrieval-Augmented Generation)
- **LLM**: OpenAI's gpt-oss:20b via Ollama
- **Embeddings**: HuggingFace sentence-transformers
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyPDF for PDF parsing

---

## 📊 Features Deep Dive

### 📄 **Document Processing**
- **Supported Formats**: PDF, TXT
- **Smart Chunking**: Recursive character splitting with overlap
- **Chunk Size**: 1024 characters with 64 character overlap
- **Batch Processing**: Handle multiple documents simultaneously

### 🧠 **AI Capabilities**
- **Question Answering**: Natural language queries about document content
- **Context Awareness**: Maintains conversation history
- **Source Attribution**: Answers based on uploaded document content
- **Fallback Handling**: Graceful error handling for edge cases

### 🎨 **User Interface**
- **Drag & Drop**: Easy document upload
- **Real-time Status**: Connection and processing status display
- **Streaming Responses**: Character-by-character response display
- **Custom Styling**: Orange-themed professional interface
- **Responsive Design**: Works on different screen sizes

---

## 🔒 Privacy & Security

- ✅ **100% Local Processing**: Documents never leave your machine
- ✅ **No External APIs**: No data sent to third-party services
- ✅ **Open Source**: Full transparency of code and model
- ✅ **Self-Hosted**: Complete control over infrastructure
- ✅ **GDPR Compliant**: No data collection or tracking

---

## 📋 Requirements

```txt
langchain>=0.1.0
langchain-community>=0.0.20
gradio>=4.0.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
pypdf>=3.17.0
python-dotenv>=1.0.0
requests>=2.31.0
```

---

## 🚀 Usage Examples

### Basic Document Chat
1. **Upload**: Drag PDF/TXT files to the upload area
2. **Process**: Click "🚀 Process Documents" to create embeddings
3. **Ask**: Type questions like:
   - "What is the main topic of this document?"
   - "Summarize the key findings"
   - "What does it say about [specific topic]?"

### Advanced Queries
```
"Compare the arguments in section 2 and section 4"
"What evidence supports the main hypothesis?"
"List all the recommendations mentioned"
"Explain the methodology used in this research"
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. Connection Failed**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
sudo systemctl restart ollama
```

**2. Model Not Found**
```bash
# Pull the required model
ollama pull gpt-oss:20b

# List available models
ollama list
```

**3. Memory Issues**
- Ensure at least 16GB RAM
- Consider using smaller chunk sizes
- Process documents one at a time for large files

**4. Remote Connection Issues**
- Check firewall settings on GPU machine
- Verify OLLAMA_HOST environment variable
- Ensure Ollama is bound to 0.0.0.0:11434

---

## 🔄 Advanced Configuration

### Custom Embeddings
```python
# In rag_langchain.py, modify create_db function:
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Better quality
)
```

### Chunk Size Optimization
```python
# Adjust in load_doc function:
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,    # Smaller for better precision
    chunk_overlap=128  # More overlap for better context
)
```

### Temperature Control
```python
# In initialize_chatbot function:
llm = Ollama(
    model=MODEL_NAME,
    base_url=OLLAMA_HOST,
    temperature=0.1,  # Lower for more focused responses
)
```

---

## 🌟 Why This Matters

### **The Open Source Revolution**
This application represents a significant milestone:

1. **First OpenAI Open Source Model in ~5 Years**: Breaking new ground since GPT-2
2. **Complete Privacy**: Your documents stay on your hardware
3. **Zero Ongoing Costs**: No API fees or subscriptions
4. **Full Customization**: Modify behavior and capabilities
5. **Educational Value**: Learn RAG concepts hands-on

### **Real-World Applications**
- **Research**: Analyze academic papers and reports
- **Legal**: Review contracts and legal documents
- **Business**: Process internal documents and manuals
- **Education**: Create interactive learning materials
- **Personal**: Organize and query personal document collections

---

## 🤝 Contributing

Contributions are welcome! This project showcases cutting-edge open-source AI capabilities.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **OpenAI** for releasing their first open-source model in ~5 years
- **Ollama** for making local LLM deployment effortless
- **LangChain** for the powerful RAG framework
- **Gradio** for the beautiful web interface
- **HuggingFace** for embeddings and model hosting
- **Facebook AI** for FAISS vector search

---

## 📞 Support

### Getting Help
1. **Connection Issues**: Check Ollama service status
2. **Model Problems**: Verify gpt-oss:20b is downloaded
3. **Performance**: Consider GPU acceleration for large documents
4. **Dependencies**: Ensure all requirements are installed

### Performance Tips
- Use SSD storage for faster document processing
- Increase RAM for larger document collections
- Consider GPU acceleration for embedding generation
- Process documents in smaller batches if memory constrained

---

**🎉 Welcome to the future of private, local document intelligence powered by OpenAI's groundbreaking open-source model!**

---

*Last updated: August 2025 - Celebrating OpenAI's historic return to open source!*
