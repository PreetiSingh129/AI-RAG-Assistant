from sentence_transformers import CrossEncoder


model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, documents, top_k=3):
    pairs = []

    for doc in documents:
        pairs.append((query, doc.page_content))

    scores = model.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked_documents[:top_k]]