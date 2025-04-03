# Documentation Voice Assistant

A powerful, locally-hosted voice-enabled documentation assistant that allows you to interact with your documentation through natural language queries. The application uses local language models (via Ollama) and text-to-speech synthesis to provide both text and voice responses.

## Features

- **Documentation Processing**: Paste text directly or upload documentation files
- **Vector Search**: Uses FAISS for efficient semantic search of content
- **Local LLM Integration**: Powered by Ollama for natural language understanding
- **Voice Responses**: High-quality text-to-speech for spoken answers
- **Conversational Interface**: Chat-based interaction with chat history
- **Customizable Voice**: Adjust speed and clarity of speech

## Installation

### Prerequisites

- Python 3.9+ installed
- [Ollama](https://ollama.com/download) installed and running
- Sufficient disk space for models (2-5GB)

### Setup

1. **Clone the repository or download the files**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:

   For Windows users, run the install script:
   ```bash
   install.bat
   ```

   Or install dependencies manually:
   ```bash
   pip install streamlit==1.34.0 streamlit_chat==0.1.1 python-dotenv==1.0.1 nest-asyncio==1.6.0
   pip install langchain==0.1.11 langchain_community==0.0.24 langchain_core==0.1.27 faiss-cpu==1.7.4
   pip install sentence-transformers==2.5.1
   pip install unstructured==0.12.4 markdown==3.5.2
   pip install numpy sounddevice scipy
   pip install pyOpenSSL==24.0.0
   pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
   pip install TTS==0.22.0
   pip install aiohttp Pillow
   ```

4. **Download LLM models with Ollama**:
   ```bash
   ollama pull deepseek-r1:7b
   ollama pull llama3:8b
   ```
   (Make sure Ollama is running)

5. **Download better TTS models** (recommended):
   ```bash
   python upgrade_tts.py
   ```
   This script will download higher-quality voice models.

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Load your documentation**:
   - Paste documentation text directly into the sidebar text area
   - Or upload a text/markdown file using the file uploader

3. **Configure settings**:
   - Select the LLM model (deepseek-r1:7b, llama3:8b, or mistral:7b)
   - Enable/disable text-to-speech
   - Adjust speech speed if TTS is enabled

4. **Initialize the system**:
   - Click the "Initialize System" button to process the documentation

5. **Ask questions**:
   - Type your question in the input field at the bottom
   - Click the "Ask" button
   - Receive both text and voice responses (if enabled)

## Troubleshooting

### Installation Issues

- **Dependency errors**: Try installing packages individually or with the `--no-deps` flag
- **PyTorch issues**: If you have a CUDA-compatible GPU, you might want to install the CUDA version of PyTorch
- **TTS installation problems**: Make sure you have all the prerequisite packages (numpy, scipy, etc.)

### Runtime Issues

- **Ollama connection errors**: Ensure Ollama is running (`http://localhost:11434`)
- **TTS voice quality**: Run `upgrade_tts.py` to download better voice models
- **Out of memory**: Reduce document size or use a smaller LLM model
- **Voice not working**: Check console for errors; voice synthesis requires additional resources

## Custom Voice Samples

For the best voice quality with XTTS:

1. Create a `speakers` directory in your project folder
2. Add WAV files for reference voices:
   - `en_female.wav` - Female English speaker sample
   - `en_male.wav` - Male English speaker sample

## Extending the Application

### Adding PDF Support

Install additional dependencies:
```bash
pip install pypdf
```

### Using GPU Acceleration

For TTS voice synthesis with GPU:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## License

This project is released under the MIT License.

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM hosting
- [Coqui TTS](https://github.com/coqui-ai/TTS) for text-to-speech capabilities
- [LangChain](https://langchain.readthedocs.io/) for the language processing framework
- [Streamlit](https://streamlit.io/) for the web interface
