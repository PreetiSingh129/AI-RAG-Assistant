# AI RAG Assistant

A small Streamlit application for querying PDF and documentation content using a Retrieval-Augmented Generation (RAG) workflow.

## Features

- Upload a PDF and ask questions about it
- Provide a documentation URL to retrieve content from the web
- Split documents into chunks and create embeddings
- Run semantic retrieval and reranking for better matches
- Generate answers from the retrieved context using Groq LLMs
- Show source references for transparency

## Project structure

- `app.py` — Streamlit UI and orchestration
- `loaders.py` — document loading helpers for PDF and URL sources
- `rag.py` — standalone script for testing the retrieval pipeline
- `reranker.py` — reranking logic using a cross-encoder model
- `requirements.txt` — project dependencies
- `data/sample.pdf` — sample PDF used for testing

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Groq API key:

   ```env
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Notes

- The app defaults to the Groq model `openai/gpt-oss-120b` because it is available on the current Groq account configuration.
- If a configured model becomes unavailable, the app falls back to an accessible model automatically.
- The project is prepared for local development and can be pushed to GitHub for deployment or sharing.

## Example use

- Upload `data/sample.pdf`
- Ask a question such as: "What programming languages does Priti know?"
- The app will retrieve relevant chunks, rerank them, and generate an answer using only the retrieved context.
