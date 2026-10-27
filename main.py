
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -------------------------------
# 1. Configuration
# -------------------------------
PDF_PATH = "document loaders/deeplearning.pdf"
DB_PATH = "chroma_db_minilm"

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

if not Path(PDF_PATH).is_file():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

# -------------------------------
# 2. Load PDF
# -------------------------------
print("Loading PDF...")

documents = PyPDFLoader(PDF_PATH).load()
print(f"Loaded {len(documents)} pages.")

if not documents:
    raise ValueError("The PDF contains no readable pages.")

# -------------------------------
# 3. Split into chunks
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

# -------------------------------
# 4. Initialize local embeddings
# -------------------------------
print("Loading MiniLM embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# 5. Create Chroma database
# -------------------------------
print("Creating vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="rag_minilm",
    persist_directory=DB_PATH
)

print(f"Vector database saved in: {DB_PATH}")

# -------------------------------
# 6. Configure MMR retriever
# -------------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# -------------------------------
# 7. Initialize Groq LLM
# -------------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# -------------------------------
# 8. Prompt template
# -------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Answer using only the supplied document context.
If the answer is not present in the context, say:
"I could not find the answer in the document."

Do not invent facts."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}"""
    )
])

# -------------------------------
# 9. Question-answering loop
# -------------------------------
print("\nRAG system ready!")
print("Ask questions about your PDF.")
print("Type 0 to exit.")

while True:
    query = input("\nYou: ").strip()

    if query == "0":
        print("Goodbye!")
        break

    if not query:
        continue

    try:
        # Retrieve relevant document chunks
        docs = retriever.invoke(query)

        if not docs:
            print("\nAI: I could not find the answer in the document.")
            continue

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        # Generate the answer
        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        print(f"\nAI: {response.content}")

        # Display source pages
        pages = sorted({
            doc.metadata.get("page", -1) + 1
            for doc in docs
            if isinstance(doc.metadata.get("page"), int)
            and doc.metadata.get("page", -1) >= 0
        })

        if pages:
            print(f"Source pages: {pages}")

    except Exception as e:
        print(f"\nError: {e}")


















































# from langchain.chat_models import init_chat_model
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# load_dotenv()

# # 1. Load PDF
# data = PyPDFLoader("document loaders/deeplearning.pdf").load()
# print(f"Loaded {len(data)} pages.")

# # 2. Split PDF into smaller chunks
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks = splitter.split_documents(data)
# print(f"Created {len(chunks)} chunks.")

# # 3. Create prompt template
# template = ChatPromptTemplate.from_messages([
#     ("system", "You are an AI assistant that summarizes text. "
#                "Give a clear, concise summary of the provided text."),
#     ("human", "{data}")
# ])

# # 4. Initialize Groq model
# model = init_chat_model(
#     "openai/gpt-oss-120b",
#     temperature=0.3,
#     model_provider="groq"
# )

# # 5. Summarize each chunk separately
# summaries = []

# for i, chunk in enumerate(chunks):
#     print(f"Summarizing chunk {i + 1}/{len(chunks)}...", flush=True)

#     prompt = template.format_messages(
#         data=chunk.page_content
#     )

#     result = model.invoke(prompt)
#     summaries.append(result.content)

# # 6. Combine chunk summaries
# combined_summary = "\n\n".join(summaries)

# print("\n--- PDF SUMMARY ---\n")
# print(combined_summary)