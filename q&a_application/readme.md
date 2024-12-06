# RAG Q&A Application

A Retrieval-Augmented Generation (RAG) based Question-Answering application built with Streamlit. The project includes two versions:
- HuggingFace API version (app.py)
- Local Ollama model version (main.py)

## Features

- Multiple input formats support:
  - PDF files
  - Text files
  - DOCX files
  - CSV files
  - URLs/Web links
  - Direct text input
- FAISS vector store for efficient similarity search
- Customizable text chunking
- Clean and interactive Streamlit interface

## Requirements

```
langchain
numpy
streamlit
langchain-community
PyPDF2
unstructured
nltk
sentence-transformers
langchain-huggingface
faiss-cpu
python-docx
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. For HuggingFace version (app.py):
   - Create a .env file
   - Add your HuggingFace API key:
     ```
     HUGGINGFACEHUB_API_TOKEN=your_api_key_here
     ```

4. For Ollama version (main.py):
   - Install Ollama locally
   - Download the required model:
     ```bash
     ollama pull llama3.2:3b
     ```
   - Ensure Ollama is running on http://localhost:11434

## Usage

Run either version using Streamlit:
```bash
streamlit run app.py   # For HuggingFace version
# OR
streamlit run main.py  # For Ollama version
```

## File Structure

- `app.py`: HuggingFace API implementation
- `main.py`: Local Ollama implementation
- `requirements.txt`: Required Python packages
- `.env`: Environment variables (not included in repo)

## Note
- The HuggingFace version requires an API key
- The Ollama version requires local installation of Ollama and the model
- Both versions provide similar functionality but use different LLM backends