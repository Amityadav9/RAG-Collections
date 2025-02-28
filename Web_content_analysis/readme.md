# Web Content Analysis Tools

This repository contains two web content analysis applications built with Streamlit and LangChain. Both tools allow you to extract, analyze, and interact with web content through sophisticated AI capabilities.

## Applications

### 1. SiteInsight AI (Hybrid Version)

SiteInsight AI offers both closed-source (OpenAI) and open-source (Ollama) options for processing and analyzing web content.

**Features:**
- Website extraction using crawl4ai
- Content summarization (OpenAI or Ollama)
- Vector embeddings creation (OpenAI or Ollama)
- Conversational interface powered by a retrieval-augmented generation (RAG) system
- Choice between closed-source and open-source AI models

### 2. WebPrism AI (Open-Source Only)

WebPrism AI is a fully open-source alternative that relies exclusively on locally-run Ollama models.

**Features:**
- Website extraction using crawl4ai
- Content summarization using local Ollama models
- Vector embeddings using Ollama's snowflake-arctic-embed model
- Conversational interface powered by RAG
- Complete privacy with all processing done locally

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed locally
- Required Ollama models:
  - `deepseek-r1:7b` (for chat)
  - `qwen2.5:14b-instruct-q4_K_M` (for summarization)
  - `snowflake-arctic-embed` (for embeddings)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/web-content-analysis.git
cd web-content-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. For SiteInsight AI (hybrid version), create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. Install Ollama models:
```bash
ollama pull deepseek-r1:7b
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull snowflake-arctic-embed
```

### Requirements

```
streamlit
langchain
langchain_community
faiss-cpu
unstructured
nest_asyncio
crawl4ai
python-dotenv
```

For the hybrid version, you'll also need:
```
openai
streamlit-chat
```

## Usage

### Running SiteInsight AI (Hybrid Version)

```bash
streamlit run siteinsight.py
```

### Running WebPrism AI (Open-Source Only)

```bash
streamlit run webprism.py
```

## How to Use

1. Enter a URL to crawl/analyze
2. Wait for the content extraction to complete
3. Generate a summary of the content
4. Create embeddings/knowledge base
5. Ask questions about the content in the chat interface

## Differences Between Versions

### SiteInsight AI (Hybrid Version)
- Offers choice between OpenAI and Ollama models
- Potentially better quality summaries with OpenAI models
- Requires an OpenAI API key
- Has additional UI components from streamlit-chat

### WebPrism AI (Open-Source Only)
- Uses only Ollama models running locally
- Complete privacy - no data sent to external APIs
- No API keys required
- Simpler UI with standard Streamlit components
- Potentially faster processing depending on local hardware

## File Structure

```
├── siteinsight.py            # Hybrid version with OpenAI and Ollama options
├── webprism.py        # Open-source only version
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the UI framework
- [LangChain](https://www.langchain.com/) for the RAG components
- [Ollama](https://ollama.ai/) for local AI models
- [crawl4ai](https://github.com/ppageur/crawl4ai) for web scraping capabilities

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.