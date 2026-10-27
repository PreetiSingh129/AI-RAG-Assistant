import os

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader


def load_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        raise ValueError(f"Could not extract any text from {file_path}")
    return documents


def load_url(url):
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError("A valid HTTP or HTTPS URL is required.")

    os.environ.setdefault("USER_AGENT", "AI-RAG-Assistant/1.0")
    loader = WebBaseLoader(url)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No content could be loaded from {url}")
    return documents