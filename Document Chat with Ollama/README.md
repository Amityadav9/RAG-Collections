# Document Chat with RAG and Ollama

This project implements a document chatbot using Retrieval-Augmented Generation (RAG) with Ollama and LlamaIndex. The application allows users to upload PDF documents and engage in meaningful conversations about their content, combining the power of large language models with precise information retrieval.

## Project Overview

This application creates an interactive chat interface where users can discuss the contents of their uploaded PDF documents. It uses a RAG architecture to ensure responses are grounded in the document's actual content, making it ideal for document analysis, research, and information extraction.

## Key Features

The application provides several powerful capabilities:

- PDF document upload and processing
- Real-time chat interface with streaming responses
- Document content preview functionality
- Advanced RAG implementation using LlamaIndex
- Integration with Ollama for local LLM inference
- High-quality embeddings using BAAI/bge-large-en-v1.5
- Efficient document indexing and retrieval
- Conversation history management

## Installation

First, create a new Python environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Setup Requirements

Before running the application, ensure you have:

1. Ollama installed and running locally
2. The llama3.2 model downloaded in Ollama
3. Sufficient system resources for running embeddings and LLM inference

## Running the Application

To start the application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Using the Application

1. Document Upload
   - Use the sidebar to upload your PDF document
   - The system will automatically process and index the document
   - A preview of the first page will be shown for verification

2. Chatting with Documents
   - Type your questions in the chat input at the bottom
   - Responses will stream in real-time
   - The system will provide answers based on the document content
   - Use the "Clear ↺" button to reset the conversation

## How It Works

The application uses several key components to provide intelligent document chat:

1. Document Processing:
   - PDFs are processed using DoclingReader
   - Content is split into manageable chunks
   - Text is converted into embeddings using BAAI/bge-large-en-v1.5

2. RAG Implementation:
   - Creates a vector store index of document content
   - Uses similarity search to find relevant passages
   - Combines retrieved content with LLM generation

3. Response Generation:
   - Utilizes Ollama for local LLM inference
   - Implements custom prompting for focused answers
   - Provides streaming responses for better user experience

## Memory Management

The application implements several memory optimization features:

- Session-based document caching
- Garbage collection after chat clearance
- Efficient streaming response handling
- Temporary file management for uploads

## Customization

You can customize the application by modifying:

- The LLM model (change 'llama3.2:latest' to other Ollama models)
- Embedding model selection
- Custom prompt templates
- Response streaming settings
- UI elements through Streamlit

## Troubleshooting

Common issues and solutions:

1. Memory Issues:
   - Ensure sufficient RAM for embedding generation
   - Clear chat history regularly
   - Process smaller documents if needed

2. Performance:
   - Check Ollama is running properly
   - Verify GPU availability if applicable
   - Monitor system resources

3. Document Processing:
   - Ensure PDF files are text-based, not scanned
   - Check file permissions
   - Verify PDF file integrity

## Contributing

Feel free to contribute to this project by:

1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Sharing feedback on usability and performance

## License

This project is open-source and available under the MIT License.

## Acknowledgments

This project uses several key technologies:
- LlamaIndex for RAG implementation
- Ollama for local LLM inference
- Streamlit for the web interface
- HuggingFace for embeddings
