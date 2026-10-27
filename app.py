import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders import load_pdf, load_url
from reranker import rerank_documents

load_dotenv()
os.environ.setdefault("USER_AGENT", "AI-RAG-Assistant/1.0")


def get_llm():
    from groq import Groq
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.warning("Add your GROQ_API_KEY to the .env file to enable AI answers.")
        st.stop()

    preferred_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    try:
        return ChatGroq(model=preferred_model, temperature=0)
    except Exception:
        client = Groq(api_key=api_key)
        available_models = [model.id for model in client.models.list().data]
        fallback_model = next(
            (model for model in available_models if "gpt-oss" in model.lower() or "llama" in model.lower() or "qwen" in model.lower()),
            available_models[0] if available_models else preferred_model,
        )
        st.warning(f"The configured Groq model was unavailable; using a valid fallback instead: {fallback_model}")
        return ChatGroq(model=fallback_model, temperature=0)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 AI Documentation & PDF RAG Assistant")

st.write(
    "Upload a PDF or provide a documentation URL and ask questions. "
    "The system retrieves relevant information using RAG and generates "
    "an answer with source references."
)


# --------------------------------------------------
# DOCUMENT SOURCE
# --------------------------------------------------

input_type = st.radio(
    "Choose document source:",
    ["📄 PDF Upload", "🌐 Documentation URL"]
)


uploaded_file = None
url = None


if input_type == "📄 PDF Upload":

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

else:

    url = st.text_input(
        "Enter documentation URL",
        placeholder="https://docs.python.org/3/"
    )


# --------------------------------------------------
# QUESTION
# --------------------------------------------------

question = st.text_input(
    "💬 Ask a question about the document"
)


# --------------------------------------------------
# MAIN RAG PIPELINE
# --------------------------------------------------

if question and (uploaded_file or url):
    pdf_path = None

    try:
        with st.spinner("📄 Loading document..."):
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    pdf_path = temp_file.name
                documents = load_pdf(pdf_path)
            else:
                if not url or not url.startswith(("http://", "https://")):
                    st.warning("Please enter a valid HTTP or HTTPS URL.")
                    st.stop()
                documents = load_url(url)

        if not documents:
            st.warning("The document could not be loaded. Please try a different file or URL.")
            st.stop()

        with st.spinner("✂️ Splitting document into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

        if not chunks:
            st.warning("The document did not produce any searchable chunks.")
            st.stop()

        st.info(f"📚 Document loaded successfully — {len(chunks)} chunks created.")

        with st.spinner("🧠 Creating embeddings..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        with st.spinner("🗄️ Creating vector database..."):
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="streamlit_rag_collection",
            )

        with st.spinner("🔎 Searching relevant information..."):
            results = vectorstore.similarity_search(question, k=6)

        with st.spinner("🎯 Reranking relevant results..."):
            reranked_results = rerank_documents(question, results, top_k=3)

        context = "\n\n".join(doc.page_content for doc in reranked_results)

        sources = []
        for doc in reranked_results:
            page = doc.metadata.get("page_label")
            source = doc.metadata.get("source", url if url else uploaded_file.name)
            source_info = f"Page {page} — {source}" if page else source
            if source_info not in sources:
                sources.append(source_info)

        with st.spinner("🤖 Generating answer..."):
            llm = get_llm()
            prompt = f"""
You are a helpful AI document assistant.

Your job is to answer the user's question using ONLY the information provided in the context.

Do not use outside knowledge.

If the answer cannot be found in the context, say:

"I couldn't find this information in the provided document."

Keep the answer clear, accurate and concise.

Context:
{context}

Question:
{question}

Answer:
"""
            response = llm.invoke(prompt)

        st.subheader("🤖 Answer")
        st.write(response.content if hasattr(response, "content") else str(response))

        st.subheader("📚 Sources")
        for source in sources:
            st.write(f"📄 {source}")

    except Exception as exc:
        st.error("❌ Something went wrong.")
        st.exception(exc)
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)