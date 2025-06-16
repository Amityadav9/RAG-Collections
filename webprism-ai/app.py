import streamlit as st
import os
import asyncio
import nest_asyncio
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from streamlit_chat import message as st_message
from langchain.schema import HumanMessage, SystemMessage

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
nest_asyncio.apply()

# ---------------------------
# Initialize Session State Variables
# ---------------------------
if "url_submitted" not in st.session_state:
    st.session_state.url_submitted = False
if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "embedding_done" not in st.session_state:
    st.session_state.embedding_done = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(layout="wide", page_title="WebPrism AI")
st.title("WebPrism AI")

page = st.sidebar.selectbox("Navigation", ["Home", "Explore Web"])

if page == "Home":
    st.markdown("""
    ## Welcome to WebPrism AI
    **WebPrism AI** is a powerful, open-source web exploration tool that helps you extract valuable insights from any website. Using advanced language models and vector search technology, WebPrism AI AI allows you to have meaningful conversations about web content without leaving this interface.
    
    **Features:**
    - **Intelligent Web Extraction:** Extract and process content from any URL
    - **Smart Summarization:** Generate concise, comprehensive summaries using local AI models
    - **Vector Embeddings:** Create semantic representations of content for precise information retrieval
    - **Conversational Interface:** Ask questions and get accurate answers about the website content
    
    All features are powered by **100% open source** models running locally through Ollama, ensuring privacy and control over your data.
    
    Get started by selecting **Explore Web** from the sidebar.
    """)

elif page == "Explore Web":
    # ---------------------------
    # URL Input Form and File Upload
    # ---------------------------
    with st.form("url_form"):
        # Add tabs for URL input and file upload
        tab1, tab2 = st.tabs(["URL Input", "File Upload"])

        with tab1:
            url_input = st.text_input("Enter a URL to analyze:")
            submit_url = st.form_submit_button("Analyze Website")

        with tab2:
            uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            submit_file = st.form_submit_button("Analyze PDF")

        if submit_url and url_input:
            st.session_state.url_submitted = True
            # Reset any previous state
            st.session_state.extraction_done = False
            st.session_state.embedding_done = False
            st.session_state.chat_history = []
            st.session_state.summary = ""

        if submit_file and uploaded_file is not None:
            st.session_state.url_submitted = True
            # Reset any previous state
            st.session_state.extraction_done = False
            st.session_state.embedding_done = False
            st.session_state.chat_history = []
            st.session_state.summary = ""

            # Save the uploaded file temporarily
            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Process the uploaded file
            try:
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader("temp_upload.pdf")
                documents = loader.load()
                # Convert documents to markdown format
                markdown_text = "\n\n".join([doc.page_content for doc in documents])
                st.session_state.extracted_text = markdown_text
                st.session_state.extraction_done = True
                st.success("PDF processing complete!")

                # Clean up the temporary file
                if os.path.exists("temp_upload.pdf"):
                    os.remove("temp_upload.pdf")

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.extraction_done = False

    # ---------------------------
    # If URL has been submitted, divide layout into three columns
    # ---------------------------
    if st.session_state.url_submitted:
        col1, col2, col3 = st.columns(3)

        # ---------------------------
        # Column 1: Website Extraction & Summarization using crawl4ai
        # ---------------------------
        with col1:
            st.header("1. Content Extraction")
            if not st.session_state.extraction_done and not uploaded_file:
                with st.spinner("Extracting website content..."):
                    # Define async crawl function (returns markdown output)
                    async def simple_crawl(url):
                        crawler_run_config = CrawlerRunConfig(
                            cache_mode=CacheMode.BYPASS
                        )

                        # Add handling for local files
                        if url.startswith("file://"):
                            try:
                                # Remove 'file://' prefix and decode URL encoding
                                local_file_path = url[7:].replace("%20", " ")
                                if not os.path.exists(local_file_path):
                                    raise FileNotFoundError(
                                        f"Local file not found: {local_file_path}"
                                    )

                                # For PDF files, we should use a PDF loader instead of direct crawling
                                if local_file_path.lower().endswith(".pdf"):
                                    from langchain_community.document_loaders import (
                                        PyPDFLoader,
                                    )

                                    loader = PyPDFLoader(local_file_path)
                                    documents = loader.load()
                                    # Convert documents to markdown format
                                    markdown_text = "\n\n".join(
                                        [doc.page_content for doc in documents]
                                    )
                                    return markdown_text

                                # For other file types, proceed with normal crawling
                                async with AsyncWebCrawler() as crawler:
                                    result = await crawler.arun(
                                        url=url, config=crawler_run_config
                                    )
                                    return result.markdown
                            except Exception as e:
                                st.error(f"Error processing local file: {str(e)}")
                                return None
                        else:
                            # Handle web URLs as before
                            async with AsyncWebCrawler() as crawler:
                                result = await crawler.arun(
                                    url=url, config=crawler_run_config
                                )
                                return result.markdown

                    # Run the async crawl (nest_asyncio makes this safe in Streamlit)
                    extracted = asyncio.run(simple_crawl(url_input))
                    if extracted is None:
                        st.error(
                            "Failed to extract content. Please check the URL or file path."
                        )
                        st.session_state.extraction_done = False
                    else:
                        st.session_state.extracted_text = extracted
                        st.session_state.extraction_done = True
                        st.success("Extraction complete!")

            # Show a preview (first few non-empty lines)
            preview = "\n".join(
                [
                    line
                    for line in st.session_state.extracted_text.splitlines()
                    if line.strip()
                ][:5]
            )
            st.text_area("Content Preview", preview, height=150)

            # Save the full extracted text as a file and provide a download button.
            st.download_button(
                label="Download Extracted Content",
                data=st.session_state.extracted_text,
                file_name="extracted_content.txt",
                mime="text/plain",
            )

            st.markdown("---")
            st.subheader("Generate Summary")

            if st.button("Summarize Content", key="summarize_button"):
                with st.spinner("Generating summary..."):
                    # German system prompt for summarization
                    system_prompt = """Sie sind ein KI-Assistent, der damit beauftragt ist, Inhalte zusammenzufassen.
Ihre Zusammenfassung sollte detailliert sein und alle wichtigen Punkte des Inhalts abdecken.
Bitte erstellen Sie eine umfassende und detaillierte Zusammenfassung im Markdown-Format.
Achten Sie besonders auf:
- Hauptthemen und Kernaussagen
- Wichtige Details und Fakten
- Struktur und Zusammenhänge
- Fachbegriffe und deren Erklärungen"""

                    # Use Ollama for summarization
                    ollama_chat = ChatOllama(
                        model="gemma3:27b",
                        base_url="http://local_host",
                        temperature=0.3,
                    )

                    # Create properly formatted messages for Ollama
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=f"Bitte fassen Sie den folgenden Inhalt zusammen:\n\n{st.session_state.extracted_text}"
                        ),
                    ]

                    # Get response
                    try:
                        response = ollama_chat(messages)
                        st.session_state.summary = response.content
                    except Exception as e:
                        st.error(f"Fehler bei der Zusammenfassung: {str(e)}")
                        st.session_state.summary = (
                            f"Fehler bei der Zusammenfassung: {str(e)}"
                        )
                st.success("Zusammenfassung erfolgreich erstellt!")

            if st.session_state.summary:
                st.subheader("Summary")
                st.markdown(st.session_state.summary, unsafe_allow_html=False)

        # ---------------------------
        # Column 2: Creating Embeddings with FAISS
        # ---------------------------
        with col2:
            st.header("2. Knowledge Base Creation")
            if st.session_state.extraction_done and not st.session_state.embedding_done:
                if st.button("Create Knowledge Base"):
                    with st.spinner("Processing content..."):
                        try:
                            # Save extracted text to a markdown file (output.md)
                            with open("output.md", "w", encoding="utf-8") as f:
                                f.write(st.session_state.extracted_text)

                            # Load the markdown file using UnstructuredMarkdownLoader
                            loader = UnstructuredMarkdownLoader("output.md")
                            data = loader.load()

                            # Modified text splitter with better chunking parameters
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=500,  # Reduced chunk size
                                chunk_overlap=50,  # Reduced overlap
                                length_function=len,
                                separators=[
                                    "\n\n",
                                    "\n",
                                    ".",
                                    "!",
                                    "?",
                                    ",",
                                    " ",
                                    "",
                                ],  # Added more separators
                            )

                            # Split the text and filter out empty chunks
                            texts = text_splitter.split_documents(data)
                            texts = [
                                doc for doc in texts if doc.page_content.strip()
                            ]  # Filter empty chunks

                            if not texts:
                                raise ValueError(
                                    "No valid text chunks were created from the document"
                                )

                            # Create embeddings using Ollama's snowflake-arctic-embed model
                            embeddings = OllamaEmbeddings(
                                model="snowflake-arctic-embed",
                                base_url="http://local_host",
                            )
                            st.info("Using Ollama snowflake-arctic-embed model")

                            # Build a FAISS vectorstore from the documents
                            vectorstore = FAISS.from_documents(texts, embeddings)

                            # Persist the vectorstore locally (optional)
                            vectorstore.save_local("faiss_index")

                            st.session_state.vectorstore = vectorstore
                            st.session_state.embedding_done = True
                            st.success("Knowledge base created successfully!")

                        except Exception as e:
                            st.error(f"Error creating knowledge base: {str(e)}")
                            st.session_state.embedding_done = False

            elif st.session_state.embedding_done:
                st.info("Knowledge base is ready. You can now chat with this content.")

        # ---------------------------
        # Column 3: Chatbot using streamlit_chat and a Retrieval Chain
        # ---------------------------
        with col3:
            st.header("3. Chat Interface")
            if st.session_state.embedding_done:
                # Setup retrieval-based QA chain using the vectorstore
                vectorstore = st.session_state.vectorstore
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                prompt_template = """
Sie sind ein KI-Assistent, der Fragen basierend auf dem bereitgestellten Kontext beantwortet.
Ihre Aufgabe ist es, eine umfassende Antwort auf die gegebene Frage zu generieren,
indem Sie ausschließlich die im Kontext verfügbaren Informationen verwenden.

Kontext: {context}

Frage: {question}

<Antwort> Bitte geben Sie Ihre Antwort im Markdown-Format. 
Achten Sie auf:
- Präzise und sachliche Antworten
- Korrekte Verwendung von Fachbegriffen
- Klare Strukturierung der Informationen
- Bezugnahme auf relevante Details aus dem Kontext
</Antwort>
"""
                prompt = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )
                chain_type_kwargs = {"prompt": prompt}

                # Initialize Ollama LLM with German-focused parameters
                llm = ChatOllama(
                    model="magistral:latest",
                    base_url="http://local_host",
                    temperature=0.3,
                )

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs=chain_type_kwargs,
                    verbose=True,
                )

                # Chat interface using streamlit_chat
                user_input = st.text_input("Ihre Frage:", key="chat_input")
                if st.button("Fragen", key="send_button") and user_input:
                    try:
                        response = qa(user_input)
                        bot_answer = response["result"]
                        st.session_state.chat_history.append(
                            {"user": user_input, "bot": bot_answer}
                        )

                        # Save the chat history to a file (chat_history.txt) for this session
                        chat_file_content = "\n\n".join(
                            [
                                f"Benutzer: {chat['user']}\nBot: {chat['bot']}"
                                for chat in st.session_state.chat_history
                            ]
                        )
                        with open("chat_history.txt", "w", encoding="utf-8") as cf:
                            cf.write(chat_file_content)
                    except Exception as e:
                        st.error(f"Fehler bei der Verarbeitung Ihrer Anfrage: {str(e)}")
                        st.session_state.chat_history.append(
                            {"user": user_input, "bot": f"Fehler: {str(e)}"}
                        )

                # Display the conversation using streamlit_chat component
                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history:
                        st_message(chat["user"], is_user=True)
                        st_message(chat["bot"], is_user=False)
            else:
                st.info("Please create the knowledge base to start chatting.")
