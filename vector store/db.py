
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = [
    Document(
        page_content="python is widely used in artificial intelligence.",
        metadata={"source": "test1.txt"}
    ),
    Document(
        page_content="pandas is used for data analysis in python.",
        metadata={"source": "test2.txt"}
    ),
    Document(
        page_content="neural networks are used in deep learning.",
        metadata={"source": "test3.txt"}
    ),
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store documents in ChromaDB
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db-local"
)

result= vectorstore.similarity_search("What is used for data analysis in python?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()

docs=retriver.invoke("explain neural networks")

for d in docs:
    print(d.page_content)
    print(d.metadata)