import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders import load_pdf
from reranker import rerank_documents

load_dotenv()
os.environ.setdefault("USER_AGENT", "AI-RAG-Assistant/1.0")


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file before running this script.")

    from groq import Groq
    from langchain_groq import ChatGroq

    preferred_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=groq_api_key)
    available_models = [model.id for model in client.models.list().data]
    model_name = preferred_model if preferred_model in available_models else next(
        (model for model in available_models if "gpt-oss" in model.lower() or "llama" in model.lower() or "qwen" in model.lower()),
        available_models[0] if available_models else preferred_model,
    )

    print(f"Using Groq model: {model_name}")

    documents = load_pdf("data/sample.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    print("Original documents:", len(documents))
    print("Total chunks:", len(chunks))
    print("\nFirst chunk:\n")
    print(chunks[0].page_content)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="my_rag_collection",
    )

    print("Chunks stored in ChromaDB successfully!")

    query = "What programming languages does Priti know?"
    results = vectorstore.similarity_search(query, k=6)
    reranked_results = rerank_documents(query, results, top_k=3)

    print("\nReranked Documents:\n")
    for i, doc in enumerate(reranked_results, 1):
        print(f"\n--- Reranked Result {i} ---")
        print(doc.page_content)
        print("Metadata:", doc.metadata)

    context = "\n\n".join(doc.page_content for doc in reranked_results)
    sources = []
    for doc in reranked_results:
        page = doc.metadata.get("page_label", doc.metadata.get("page", "Unknown"))
        source = doc.metadata.get("source", "Unknown")
        source_info = f"Page {page} — {source}"
        if source_info not in sources:
            sources.append(source_info)

    llm = ChatGroq(model=model_name, temperature=0)
    prompt = f"""
You are a helpful document assistant.

Answer the question using ONLY the information provided in the context.

If the answer cannot be found in the context, say:
"I couldn't find this information in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)

    print("\nFinal Answer:\n")
    print(response.content)
    print("\nSources:\n")
    for source in sources:
        print(f"- {source}")


if __name__ == "__main__":
    main()

