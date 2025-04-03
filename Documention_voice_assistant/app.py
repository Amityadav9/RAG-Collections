from typing import List, Dict, Optional
import os
import uuid
import tempfile
import asyncio
import sys
import logging

import streamlit as st
from dotenv import load_dotenv
import numpy as np
from streamlit_chat import message as st_message

# For vector storage and retrieval
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# For Text-to-Speech
from TTS.api import TTS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Initialize session state variables
def init_session_state():
    defaults = {
        "initialized": False,
        "doc_text": "",
        "setup_complete": False,
        "vector_store": None,
        "chat_history": [],
        "selected_voice": "en_female",  # Default voice
        "llm_model": "deepseek-r1:7b",  # Default LLM model
        "tts_enabled": True,
        "tts_model": None,
        "voice_speed": 1.0,
        "current_query": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sidebar_config():
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")

        # Text area for pasting documentation
        st.subheader("Input Documentation")
        st.session_state.doc_text = st.text_area(
            "Paste documentation here",
            value=st.session_state.doc_text,
            height=300,
            placeholder="Paste your documentation text here...",
        )

        # File uploader for documentation files
        uploaded_file = st.file_uploader(
            "Or upload a text/markdown file", type=["txt", "md"]
        )
        if uploaded_file is not None:
            # Read file content
            file_content = uploaded_file.getvalue().decode("utf-8")
            st.session_state.doc_text = file_content
            st.success(
                f"Loaded {len(file_content)} characters from {uploaded_file.name}"
            )

        st.markdown("---")
        st.markdown("### 🧠 Model Settings")

        llm_models = ["deepseek-r1:7b", "llama3.2:3b", "gemma3:12b"]
        st.session_state.llm_model = st.selectbox(
            "Select LLM Model",
            options=llm_models,
            index=llm_models.index(st.session_state.llm_model)
            if st.session_state.llm_model in llm_models
            else 0,
            help="Choose the local LLM model (requires Ollama)",
        )

        st.markdown("### 🎤 Voice Settings")
        st.session_state.tts_enabled = st.checkbox(
            "Enable Text-to-Speech", value=st.session_state.tts_enabled
        )

        # Add voice speed option
        if "voice_speed" not in st.session_state:
            st.session_state.voice_speed = 1.0

        if st.session_state.tts_enabled:
            st.session_state.voice_speed = st.slider(
                "Speech Speed",
                min_value=0.7,
                max_value=1.3,
                value=st.session_state.voice_speed,
                step=0.1,
                help="Adjust the speed of the generated speech",
            )

        if st.button("Initialize System", type="primary"):
            if st.session_state.doc_text:
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    try:
                        st.markdown("🔄 Processing documentation...")
                        process_documentation(progress_placeholder)
                    except Exception as e:
                        st.error(f"Error during setup: {str(e)}")
            else:
                st.error("Please paste or upload documentation text!")


def process_documentation(progress_placeholder):
    try:
        # Get content from the text area
        content = st.session_state.doc_text

        # Process the content into chunks and create vector store
        progress_placeholder.markdown("🔄 Processing content into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        chunks = text_splitter.split_text(content)
        progress_placeholder.markdown(f"✅ Split content into {len(chunks)} chunks")

        # Create embeddings
        progress_placeholder.markdown("🔄 Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create FAISS vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        st.session_state.vector_store = vector_store

        # Initialize TTS if enabled
        if st.session_state.tts_enabled:
            progress_placeholder.markdown("🔄 Initializing TTS model...")
            tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
            st.session_state.tts_model = tts
            progress_placeholder.markdown("✅ TTS model initialized!")

        progress_placeholder.markdown("✅ Vector store created successfully!")
        st.session_state.setup_complete = True
        progress_placeholder.success("System initialization complete!")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        progress_placeholder.error(f"Error: {str(e)}")
        raise e


def process_query(query):
    try:
        if not st.session_state.setup_complete or not st.session_state.vector_store:
            return {
                "status": "error",
                "error": "System not initialized. Please initialize the system first.",
            }

        # Retrieve relevant chunks from vector store
        docs = st.session_state.vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Initialize Ollama chat model
        llm = ChatOllama(
            model=st.session_state.llm_model,
            base_url="http://localhost:11434",
            temperature=0.2,
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful documentation assistant. Your task is to:
            1. Analyze the provided documentation content
            2. Answer the user's question clearly and concisely
            3. Include relevant examples when available
            4. Keep responses natural and conversational
            5. Format your response in a way that's easy to read and understand
            
            Use only the information provided in the context to answer the question.
            If you don't know the answer based on the provided context, say so honestly.""",
                ),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        # Set up the chain
        chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Execute the chain
        response = chain.invoke({"context": context, "question": query})

        # Update chat history
        st.session_state.chat_history.append({"user": query, "bot": response})

        # Generate TTS if enabled
        audio_path = None
        if st.session_state.tts_enabled and st.session_state.tts_model:
            try:
                # Generate audio with improved clarity
                temp_dir = tempfile.gettempdir()
                audio_path = os.path.join(temp_dir, f"response_{uuid.uuid4()}.wav")

                # Choose a better speaker for clearer voice
                tts_model = st.session_state.tts_model

                # Find best speaker (p225 and p226 are often clearer female voices)
                best_speaker = None
                for speaker in tts_model.speakers:
                    if speaker in [
                        "p225",
                        "p226",
                        "p270",
                        "p271",
                    ]:  # Prioritize these voices for clarity
                        best_speaker = speaker
                        break

                if not best_speaker and tts_model.speakers:
                    best_speaker = tts_model.speakers[0]

                # Break response into smaller chunks for better speech synthesis
                # This improves clarity by allowing better phrasing
                sentences = response.replace("\n", " ").split(". ")
                chunks = []
                current_chunk = ""

                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    if (
                        len(current_chunk) + len(sentence) < 200
                    ):  # Optimal chunk size for TTS
                        current_chunk += sentence + ". "
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence + ". "

                if current_chunk:
                    chunks.append(current_chunk)

                # Process each chunk with TTS
                temp_audio_files = []
                for i, chunk in enumerate(chunks):
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}_{uuid.uuid4()}.wav")
                    tts_model.tts_to_file(
                        text=chunk, file_path=chunk_path, speaker=best_speaker
                    )
                    temp_audio_files.append(chunk_path)

                # Combine audio files if multiple chunks
                if len(temp_audio_files) > 1:
                    import wave

                    data = []
                    for file in temp_audio_files:
                        w = wave.open(file, "rb")
                        data.append([w.getparams(), w.readframes(w.getnframes())])
                        w.close()

                    output = wave.open(audio_path, "wb")
                    output.setparams(data[0][0])
                    for i in range(len(data)):
                        output.writeframes(data[i][1])
                    output.close()

                    # Clean up temp files
                    for file in temp_audio_files:
                        try:
                            os.remove(file)
                        except:
                            pass
                else:
                    # Just use the single file
                    audio_path = temp_audio_files[0]

            except Exception as e:
                logger.error(f"Error generating TTS: {str(e)}")
                st.warning(f"Could not generate audio: {str(e)}")

        return {
            "status": "success",
            "text_response": response,
            "audio_path": audio_path,
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"status": "error", "error": str(e), "query": query}


def run_streamlit():
    st.set_page_config(
        page_title="Documentation Voice Assistant", page_icon="🎙️", layout="wide"
    )

    init_session_state()
    sidebar_config()

    st.title("🎙️ Documentation Assistant")
    st.markdown("""
    Get answers to your documentation questions! Simply:
    1. Paste your documentation text in the sidebar
    2. Initialize the system to process the content
    3. Ask your questions in the chat interface below
    
    This system uses:
    - Ollama for LLM inference
    - FAISS for vector search
    - TTS for voice synthesis (if enabled)
    """)

    # Create a container for the chat interface
    chat_container = st.container()

    # Create a container for the input field at the bottom of the page
    input_container = st.container()

    # Display the chat history
    with chat_container:
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                st_message(chat["user"], is_user=True, key=f"user_{i}")
                st_message(chat["bot"], is_user=False, key=f"bot_{i}")

    # Input field for the user query
    with input_container:
        query = st.text_input(
            "What would you like to know about the documentation?",
            placeholder="e.g., How do I authenticate API requests?",
            disabled=not st.session_state.setup_complete,
            key="query_input",
        )

        # Add a submit button next to the text input
        col1, col2 = st.columns([5, 1])
        with col1:
            submit_button = st.button(
                "Ask", disabled=not st.session_state.setup_complete
            )

        if (submit_button or query) and st.session_state.setup_complete and query:
            # Store the current query
            current_query = query

            # Process the query
            result = process_query(current_query)

            if result["status"] == "success":
                # Display audio if available
                if result.get("audio_path") and os.path.exists(result["audio_path"]):
                    st.audio(result["audio_path"], format="audio/wav")

                # Store query in session state to preserve it across reruns
                if "last_query" not in st.session_state:
                    st.session_state.last_query = current_query

                # Force a rerun to update the UI with the new chat messages
                st.rerun()
            else:
                st.error(f"Error: {result.get('error', 'Unknown error occurred')}")

    if not st.session_state.setup_complete:
        st.info(
            "👈 Please paste documentation and initialize the system using the sidebar first!"
        )


if __name__ == "__main__":
    run_streamlit()
