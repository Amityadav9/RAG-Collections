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
    # URL Input Form
    # ---------------------------
    with st.form("url_form"):
        url_input = st.text_input("Enter a URL to analyze:")
        submit_url = st.form_submit_button("Analyze Website")
        if submit_url and url_input:
            st.session_state.url_submitted = True
            # Reset any previous state
            st.session_state.extraction_done = False
            st.session_state.embedding_done = False
            st.session_state.chat_history = []
            st.session_state.summary = ""

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
            if not st.session_state.extraction_done:
                with st.spinner("Extracting website content..."):
                    # Define async crawl function (returns markdown output)
                    async def simple_crawl(url):
                        crawler_run_config = CrawlerRunConfig(
                            cache_mode=CacheMode.BYPASS
                        )
                        async with AsyncWebCrawler() as crawler:
                            result = await crawler.arun(
                                url=url, config=crawler_run_config
                            )
                            return result.markdown

                    # Run the async crawl (nest_asyncio makes this safe in Streamlit)
                    extracted = asyncio.run(simple_crawl(url_input))
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
                    # Common summary system prompt
                    system_prompt = """You are an AI assistant that is tasked with summarizing a web page.
Your summary should be detailed and cover all key points mentioned in the web page.
 Below is the extracted content of the web page:
{content}

Please provide a comprehensive and detailed summary in Markdown format.
"""

                    # Use Ollama for summarization
                    ollama_chat = ChatOllama(
                        model="qwen2.5:14b-instruct-q4_K_M",
                        base_url="http://localhost:11434",
                        temperature=0.3,
                    )

                    # Create properly formatted messages for Ollama
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=f"Below is the extracted content of the web page:\n\n{st.session_state.extracted_text}\n\nPlease provide a comprehensive and detailed summary in Markdown format."
                        ),
                    ]

                    # Get response
                    try:
                        response = ollama_chat(messages)
                        st.session_state.summary = response.content
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                        st.session_state.summary = f"Error generating summary: {str(e)}"
                st.success("Summary generation complete!")

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
                        # Save extracted text to a markdown file (output.md)
                        with open("output.md", "w", encoding="utf-8") as f:
                            f.write(st.session_state.extracted_text)

                        # Load the markdown file using UnstructuredMarkdownLoader
                        loader = UnstructuredMarkdownLoader("output.md")
                        data = loader.load()

                        # Split the text into chunks using RecursiveCharacterTextSplitter
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000, chunk_overlap=100
                        )
                        texts = text_splitter.split_documents(data)

                        # Create embeddings using Ollama's snowflake-arctic-embed model
                        embeddings = OllamaEmbeddings(
                            model="snowflake-arctic-embed",
                            base_url="http://localhost:11434",
                        )
                        st.info("Using Ollama snowflake-arctic-embed model")

                        # Build a FAISS vectorstore from the documents
                        vectorstore = FAISS.from_documents(texts, embeddings)

                        # Persist the vectorstore locally (optional)
                        vectorstore.save_local("faiss_index")

                        st.session_state.vectorstore = vectorstore
                        st.session_state.embedding_done = True
                    st.success("Knowledge base created successfully!")
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
You are an AI assistant tasked with answering questions based solely
on the provided context. Your goal is to generate a comprehensive answer
for the given question using only the information available in the context.

context: {context}

question: {question}

<response> Your answer in Markdown format. </response>
"""
                prompt = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )
                chain_type_kwargs = {"prompt": prompt}

                # Initialize Ollama LLM
                llm = ChatOllama(
                    model="deepseek-r1:7b",
                    base_url="http://localhost:11434",
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
                user_input = st.text_input("Your Question:", key="chat_input")
                if st.button("Ask", key="send_button") and user_input:
                    try:
                        response = qa(user_input)
                        bot_answer = response["result"]
                        st.session_state.chat_history.append(
                            {"user": user_input, "bot": bot_answer}
                        )

                        # Save the chat history to a file (chat_history.txt) for this session
                        chat_file_content = "\n\n".join(
                            [
                                f"User: {chat['user']}\nBot: {chat['bot']}"
                                for chat in st.session_state.chat_history
                            ]
                        )
                        with open("chat_history.txt", "w", encoding="utf-8") as cf:
                            cf.write(chat_file_content)
                    except Exception as e:
                        st.error(f"Error processing your query: {str(e)}")
                        st.session_state.chat_history.append(
                            {"user": user_input, "bot": f"Error: {str(e)}"}
                        )

                # Display the conversation using streamlit_chat component
                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history:
                        st_message(chat["user"], is_user=True)
                        st_message(chat["bot"], is_user=False)
            else:
                st.info("Please create the knowledge base to start chatting.")
