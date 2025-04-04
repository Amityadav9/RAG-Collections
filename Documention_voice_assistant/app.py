from typing import List, Dict, Optional
import os
import uuid
import tempfile
import asyncio
import sys
import logging
import torch
import time
from threading import Lock

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

# Set up logging - redirect to file instead of console
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global lock to prevent concurrent processing
query_lock = Lock()

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
    )
else:
    logger.info("Using CPU for processing")


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
        "processing": False,
        "progress_status": "",
        "query_submitted": False,
        "temp_query_value": "",  # Use this for storing the query value temporarily
        "should_clear_input": False,  # Flag to indicate input should be cleared
        "last_response": None,
        "audio_data": None,  # Store audio data
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

            # Add TTS model selection
            tts_models = ["tts_models/en/vctk/vits"]

            if "tts_model_name" not in st.session_state:
                st.session_state.tts_model_name = tts_models[0]

            st.session_state.tts_model_name = st.selectbox(
                "TTS Model",
                options=tts_models,
                index=tts_models.index(st.session_state.tts_model_name)
                if st.session_state.tts_model_name in tts_models
                else 0,
                help="Select the Text-to-Speech model",
            )

            # GPU usage for TTS
            if DEVICE == "cuda":
                if "tts_use_gpu" not in st.session_state:
                    st.session_state.tts_use_gpu = True

                st.session_state.tts_use_gpu = st.checkbox(
                    "Use GPU for TTS",
                    value=st.session_state.tts_use_gpu,
                    help="Enable GPU acceleration for Text-to-Speech (recommended)",
                )

        # System initialization button
        if st.button("Initialize System", type="primary"):
            if st.session_state.doc_text:
                st.session_state.processing = True
                st.session_state.progress_status = "Processing documentation..."
                st.rerun()
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

        # Create embeddings - use GPU if available
        progress_placeholder.markdown("🔄 Creating embeddings...")
        model_kwargs = {"device": DEVICE} if DEVICE == "cuda" else {}
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs
        )

        # Create FAISS vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        st.session_state.vector_store = vector_store

        # Initialize TTS if enabled
        if st.session_state.tts_enabled:
            progress_placeholder.markdown("🔄 Initializing TTS model...")

            # Get selected TTS model name
            model_name = st.session_state.tts_model_name

            # Initialize TTS with GPU if available and enabled
            use_gpu = DEVICE == "cuda" and st.session_state.tts_use_gpu
            tts = TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)
            st.session_state.tts_model = tts
            progress_placeholder.markdown(f"✅ TTS model initialized ({model_name})")

            # Show available speakers for this model
            if hasattr(tts, "speakers") and tts.speakers:
                if len(tts.speakers) <= 10:
                    speaker_list = ", ".join(tts.speakers)
                else:
                    speaker_list = (
                        ", ".join(tts.speakers[:10])
                        + f" and {len(tts.speakers) - 10} more"
                    )
                progress_placeholder.markdown(f"Available speakers: {speaker_list}")

        progress_placeholder.markdown("✅ Vector store created successfully!")
        st.session_state.setup_complete = True
        progress_placeholder.success("System initialization complete!")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        progress_placeholder.error(f"Error: {str(e)}")
        raise e


def process_query(query):
    # Use lock to prevent concurrent processing
    if not query_lock.acquire(blocking=False):
        logger.info("Another query is already being processed")
        return {
            "status": "error",
            "error": "Another query is already being processed. Please wait.",
        }

    try:
        if not st.session_state.setup_complete or not st.session_state.vector_store:
            query_lock.release()
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

        # Update chat history if this is a new response
        st.session_state.chat_history.append({"user": query, "bot": response})
        st.session_state.last_response = response

        # Generate TTS if enabled
        audio_data = None
        if st.session_state.tts_enabled and st.session_state.tts_model:
            try:
                # Generate audio
                temp_dir = tempfile.gettempdir()
                audio_path = os.path.join(temp_dir, f"response_{uuid.uuid4()}.wav")

                tts_model = st.session_state.tts_model

                # Find best speaker
                best_speaker = None
                preferred_speakers = ["p225", "p226", "p270", "p271"]

                for speaker in preferred_speakers:
                    if speaker in tts_model.speakers:
                        best_speaker = speaker
                        break

                if not best_speaker and tts_model.speakers:
                    best_speaker = tts_model.speakers[0]

                # Generate speech with selected speaker
                logger.info(f"Generating speech with speaker {best_speaker}")
                tts_model.tts_to_file(
                    text=response[:1000],  # Limit length to prevent long processing
                    file_path=audio_path,
                    speaker=best_speaker,
                    speed=st.session_state.voice_speed,
                )

                # Read the audio file
                with open(audio_path, "rb") as f:
                    audio_data = f.read()

                # Store audio data in session state
                st.session_state.audio_data = audio_data
                logger.info(f"Audio generated successfully: {len(audio_data)} bytes")

            except Exception as e:
                logger.error(f"Error generating TTS: {str(e)}")
                st.warning(f"Could not generate audio: {str(e)}")

        # Set flag to clear input on next render
        st.session_state.should_clear_input = True

        return {
            "status": "success",
            "text_response": response,
            "audio_data": audio_data,
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"status": "error", "error": str(e), "query": query}
    finally:
        # Always release the lock
        query_lock.release()


def handle_query_input():
    # Save the current query to process
    if st.session_state.query_input:
        st.session_state.temp_query_value = st.session_state.query_input
        st.session_state.query_submitted = True


def run_streamlit():
    st.set_page_config(
        page_title="Documentation Voice Assistant", page_icon="🎙️", layout="wide"
    )

    init_session_state()

    # Handle initialization processing
    if st.session_state.processing:
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            try:
                st.markdown(f"🔄 {st.session_state.progress_status}")
                process_documentation(progress_placeholder)
                st.session_state.processing = False
            except Exception as e:
                st.error(f"Error during setup: {str(e)}")
                st.session_state.processing = False

    sidebar_config()

    st.title("🎙️ Documentation Voice Assistant")

    # Show GPU status if available
    if DEVICE == "cuda":
        st.sidebar.success(f"🚀 GPU Enabled: {torch.cuda.get_device_name(0)}")

    st.markdown("""
    Get answers to your documentation questions! Simply:
    1. Paste your documentation text in the sidebar or upload a file
    2. Initialize the system to process the content
    3. Ask your questions in the chat interface below
    """)

    # Create a container for the chat interface
    chat_container = st.container()

    # Create a container for the input field at the bottom of the page
    input_container = st.container()

    # Display the chat history
    with chat_container:
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                st_message(
                    chat["user"], is_user=True, key=f"user_{i}_{hash(chat['user'])}"
                )
                st_message(
                    chat["bot"], is_user=False, key=f"bot_{i}_{hash(chat['bot'])}"
                )

                # Display audio for the latest message
                if (
                    i == len(st.session_state.chat_history) - 1
                    and st.session_state.audio_data
                ):
                    st.audio(st.session_state.audio_data, format="audio/wav")
                    # Include download button
                    st.download_button(
                        label="Download Audio Response",
                        data=st.session_state.audio_data,
                        file_name=f"response_{i}.wav",
                        mime="audio/wav",
                    )

    # Input field for the user query
    with input_container:
        # Handle clearing input if needed
        initial_value = (
            ""
            if st.session_state.should_clear_input
            else st.session_state.get("query_input", "")
        )
        if st.session_state.should_clear_input:
            st.session_state.should_clear_input = False  # Reset flag

        query = st.text_input(
            "What would you like to know about the documentation?",
            value=initial_value,
            key="query_input",
            placeholder="e.g., How do I authenticate API requests?",
            disabled=not st.session_state.setup_complete
            or st.session_state.query_submitted,
            on_change=handle_query_input,
        )

        # Add a submit button
        col1, col2 = st.columns([5, 1])
        with col2:
            submit_button = st.button(
                "Ask",
                disabled=not st.session_state.setup_complete
                or not query
                or st.session_state.query_submitted,
            )

        # Process query when the button is clicked or input is submitted
        if (
            submit_button or st.session_state.query_submitted
        ) and st.session_state.setup_complete:
            # If query was just submitted through the on_change callback
            if st.session_state.query_submitted and st.session_state.temp_query_value:
                query_to_process = st.session_state.temp_query_value

                # Show a spinner during processing
                with st.spinner("Processing your query..."):
                    # Process the query
                    result = process_query(query_to_process)

                    if result["status"] == "success":
                        # Reset submission flags
                        st.session_state.query_submitted = False
                        st.session_state.temp_query_value = ""

                        # Rerun to update UI without modifying the input directly
                        time.sleep(0.5)  # Small delay to ensure UI update
                        st.rerun()
                    else:
                        st.error(
                            f"Error: {result.get('error', 'Unknown error occurred')}"
                        )
                        st.session_state.query_submitted = False
                        st.session_state.temp_query_value = ""

    if not st.session_state.setup_complete:
        st.info(
            "👈 Please paste documentation and initialize the system using the sidebar first!"
        )


if __name__ == "__main__":
    run_streamlit()
