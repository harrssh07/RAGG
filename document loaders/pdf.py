from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data=PyPDFLoader("document loaders/GRU.pdf").load()

docs=data

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunks=splitter.split_documents(docs)


print(len(chunks))




