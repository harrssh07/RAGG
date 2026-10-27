import os
from langchain_community.document_loaders import WebBaseLoader

os.environ["USER_AGENT"] = "RAG-Learning-Project/1.0"

url = "https://docs.mistral.ai/"

loader = WebBaseLoader(
    web_paths=(url,),
    requests_kwargs={"timeout": 15},
)

docs = loader.load()

print(docs[0].page_content)
