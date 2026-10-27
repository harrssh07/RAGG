import requests
import xml.etree.ElementTree as ET
from langchain_core.documents import Document

url = "https://export.arxiv.org/api/query"

params = {
    "search_query": 'all:"large language models"',
    "start": 0,
    "max_results": 3,
    "sortBy": "relevance",
    "sortOrder": "descending",
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

root = ET.fromstring(response.text)
namespace = {"atom": "http://www.w3.org/2005/Atom"}

docs = []

for entry in root.findall("atom:entry", namespace):
    title = entry.findtext("atom:title", default="", namespaces=namespace)
    summary = entry.findtext("atom:summary", default="", namespaces=namespace)
    paper_url = entry.findtext("atom:id", default="", namespaces=namespace)

    docs.append(
        Document(
            page_content=f"Title: {title.strip()}\n\n{summary.strip()}",
            metadata={"source": paper_url},
        )
    )

for doc in docs:
    print(doc.page_content)
    print("Source:", doc.metadata["source"])
    print("-" * 60)