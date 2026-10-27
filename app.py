
import os
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
DB_ROOT = BASE_DIR / "chroma_db_minilm"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_ROOT.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==================================================
# 2. MODELS
# ==================================================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_llm(api_key, model_name):
    return ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        temperature=0,
    )


# ==================================================
# 3. FILE AND INDEX HELPERS
# ==================================================

def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def get_index_details(file_path, chunk_size, chunk_overlap):
    file_hash = calculate_file_hash(file_path)

    index_id = (
        f"{file_hash[:16]}_"
        f"{chunk_size}_{chunk_overlap}_minilm"
    )

    db_path = DB_ROOT / index_id
    collection_name = f"pdf_{index_id}".replace("-", "_")
    marker_file = db_path / "index_ready.txt"

    return db_path, collection_name, marker_file


def prepare_vectorstore(file_path, chunk_size, chunk_overlap):
    """Create embeddings only when the user requests processing."""

    db_path, collection_name, marker_file = get_index_details(
        file_path,
        chunk_size,
        chunk_overlap,
    )

    embeddings = get_embeddings()

    # Reuse an existing index if available.
    if marker_file.exists():
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(db_path),
        )

        return vectorstore, True

    # Read and split the PDF.
    documents = PyPDFLoader(str(file_path)).load()

    if not documents:
        raise ValueError("No readable pages were found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be created.")

    for chunk in chunks:
        chunk.metadata["source_file"] = Path(file_path).name

    db_path.mkdir(parents=True, exist_ok=True)

    # This is where embedding and indexing happen.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(db_path),
    )

    # Mark the index ready after successful creation.
    marker_file.write_text("ready", encoding="utf-8")

    return vectorstore, False


# ==================================================
# 4. PROMPT
# ==================================================

PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful PDF question-answering assistant.

Answer using only the information in the supplied context.
Explain concepts clearly and use simple language.
Do not invent information.
If the answer is not in the context, say:
"I could not find enough information about this in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:
"""
)


def format_context(documents):
    result = []

    for i, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page")

        if isinstance(page, int):
            page_label = f"Page {page + 1}"
        else:
            page_label = "Page unknown"

        result.append(
            f"[Source {i} | {page_label}]\n{doc.page_content}"
        )

    return "\n\n".join(result)


# ==================================================
# 5. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_key" not in st.session_state:
    st.session_state.processed_key = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:
    st.title("⚙️ Settings")

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if api_key:
        st.success("Groq API key loaded securely.")
    else:
        st.error("GROQ_API_KEY is missing from your .env file.")

    model_name = st.text_input(
        "Groq Model",
        value="openai/gpt-oss-120b",
    )

    st.divider()
    st.subheader("Chunking")

    chunk_size = st.number_input(
        "Chunk size",
        min_value=200,
        max_value=3000,
        value=1000,
        step=100,
    )

    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=1000,
        value=200,
        step=50,
    )

    st.divider()
    st.subheader("Retrieval")

    k = st.slider(
        "Retrieved chunks (k)",
        min_value=1,
        max_value=10,
        value=4,
    )

    fetch_k = st.slider(
        "Candidate chunks (fetch_k)",
        min_value=5,
        max_value=30,
        value=10,
    )

    lambda_mult = st.slider(
        "MMR relevance/diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
    )

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset selected PDF", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.processed_key = None
        st.session_state.processed_file = None
        st.session_state.messages = []
        st.rerun()


# ==================================================
# 7. MAIN INTERFACE
# ==================================================

st.title("📚 PDF RAG Assistant")

st.caption(
    "Upload a document, choose your settings, and process it "
    "only when you are ready."
)

uploaded_pdf = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Selecting a PDF does not start embedding.",
)

pdf_path = None
file_hash = None

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    pdf_path = UPLOAD_DIR / f"{file_hash}.pdf"

    if not pdf_path.exists():
        pdf_path.write_bytes(pdf_bytes)

    st.write(f"Selected document: **{uploaded_pdf.name}**")

else:
    st.info("Upload a PDF to get started. No document is processed automatically.")


# ==================================================
# 8. EXPLICIT PROCESS BUTTON
# ==================================================

settings_key = None

if pdf_path is not None:
    settings_key = (
        file_hash,
        int(chunk_size),
        int(chunk_overlap),
    )

    already_processed = (
        st.session_state.processed_key == settings_key
        and st.session_state.vectorstore is not None
    )

    if already_processed:
        st.success("This PDF is ready. You can ask questions below.")
    else:
        st.info(
            "Your PDF has not been processed with the current settings. "
            "Click the button below when you are ready."
        )

if chunk_overlap >= chunk_size:
    st.warning("Chunk overlap must be smaller than chunk size.")

process_clicked = st.button(
    "🔄 Process PDF",
    type="primary",
    disabled=(
        pdf_path is None
        or not api_key
        or chunk_overlap >= chunk_size
    ),
    use_container_width=False,
)

if process_clicked and pdf_path is not None:
    # Prevent an old index from being used during processing.
    st.session_state.vectorstore = None
    st.session_state.processed_key = None
    st.session_state.processed_file = None
    st.session_state.messages = []

    try:
        with st.spinner(
            "Processing PDF and preparing the vector index..."
        ):
            vectorstore, index_existed = prepare_vectorstore(
                pdf_path,
                int(chunk_size),
                int(chunk_overlap),
            )

        st.session_state.vectorstore = vectorstore
        st.session_state.processed_key = settings_key
        st.session_state.processed_file = str(pdf_path)

        if index_existed:
            st.success("Loaded existing PDF index. No new embeddings needed.")
        else:
            st.success("PDF processed and embeddings created successfully.")

    except Exception as error:
        st.error(f"PDF processing failed: {error}")
        st.stop()

    st.rerun()


# ==================================================
# 9. CHAT AVAILABILITY
# ==================================================

ready_to_chat = (
    pdf_path is not None
    and st.session_state.vectorstore is not None
    and st.session_state.processed_key == settings_key
)

if pdf_path is not None and not ready_to_chat:
    st.warning(
        "Chat is disabled until you click Process PDF "
        "for the selected document and current chunk settings."
    )


# ==================================================
# 10. DISPLAY CHAT HISTORY
# ==================================================

if ready_to_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 View retrieved sources"):
                    for i, source in enumerate(
                        message["sources"], start=1
                    ):
                        page = source.metadata.get("page")
                        page_label = (
                            f"Page {page + 1}"
                            if isinstance(page, int)
                            else "Page unknown"
                        )

                        st.markdown(f"**Source {i} — {page_label}**")
                        st.write(source.page_content)
                        st.divider()


# ==================================================
# 11. ASK QUESTIONS
# ==================================================

if ready_to_chat:
    question = st.chat_input("Ask something about your PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                llm = get_llm(api_key, model_name)

                with st.spinner("Searching the PDF..."):
                    retriever = (
                        st.session_state.vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": int(k),
                                "fetch_k": max(int(fetch_k), int(k)),
                                "lambda_mult": float(lambda_mult),
                            },
                        )
                    )

                    retrieved_docs = retriever.invoke(question)

                    if not retrieved_docs:
                        answer = (
                            "I could not find relevant information "
                            "in the uploaded PDF."
                        )
                    else:
                        chain = PROMPT | llm

                        response = chain.invoke(
                            {
                                "context": format_context(retrieved_docs),
                                "question": question,
                            }
                        )

                        answer = str(response.content)

                st.markdown(answer)

                if retrieved_docs:
                    with st.expander("📄 View retrieved sources"):
                        for i, source in enumerate(
                            retrieved_docs, start=1
                        ):
                            page = source.metadata.get("page")
                            page_label = (
                                f"Page {page + 1}"
                                if isinstance(page, int)
                                else "Page unknown"
                            )

                            st.markdown(f"**Source {i} — {page_label}**")
                            st.write(source.page_content)
                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_docs,
                    }
                )

            except Exception:
                st.error(
                    "Could not generate an answer. Check your Groq "
                    "model, API key, and terminal logs."
                )
                st.exception(Exception)
