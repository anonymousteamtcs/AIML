import os
import math
import json
import hashlib
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings

# Optional: token-level F1 helper (simple)
def _tokenize_text(s: str) -> List[str]:
    return s.lower().split()

def f1_score_tokens(pred: str, gold: str) -> float:
    p_tokens = _tokenize_text(pred)
    g_tokens = _tokenize_text(gold)
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    precision = match / len(p_tokens)
    recall = match / len(g_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# 1. Chunking and splitting of large data without loss
def chunk_texts(
    documents: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    keep_meta: bool = True
) -> List[Dict]:
    """
    documents: list of dicts with keys: 'id', 'text', optional 'meta'
    Returns list of chunks with fields: 'chunk_id', 'doc_id', 'text', 'meta'
    Uses LangChain RecursiveCharacterTextSplitter to preserve text and deterministic splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for doc in documents:
        doc_id = doc.get("id") or hashlib.sha1(doc.get("text","").encode()).hexdigest()
        text = doc.get("text", "")
        meta = doc.get("meta", {})
        split_texts = splitter.split_text(text)
        for i, t in enumerate(split_texts):
            chunk = {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "text": t,
                "meta": meta if keep_meta else {}
            }
            chunks.append(chunk)
    return chunks

# 2. Create embeddings
def create_embeddings(
    texts: List[str],
    openai_api_key: Optional[str] = None,
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    Returns list of embeddings (floats) for each text.
    Uses LangChain OpenAIEmbeddings wrapper.
    Provide API key via env or openai_api_key param.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    embedder = OpenAIEmbeddings(model=model)
    # LangChain handles batching internally; return as list of lists
    embeddings = embedder.embed_documents(texts)
    return embeddings

# 3. Store in vector DB (Chroma)
def store_in_chroma(
    collection_name: str,
    chunks: List[Dict],
    embeddings: List[List[float]],
    persist_directory: str = "./chromadb_store",
    distance_metric: str = "cosine"
) -> chromadb.api.models.Collection.Collection:
    """
    Stores chunk vectors + metadata in a local chroma collection.
    Expects len(chunks) == len(embeddings).
    Each chunk dict should contain chunk_id, text, doc_id, meta.
    Returns chroma collection object.
    """
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    # Create or get collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name, metadata={"distance": distance_metric})

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"doc_id": c["doc_id"], **(c.get("meta") or {})} for c in chunks]
    documents = [c["text"] for c in chunks]

    # Upsert - chroma will overwrite duplicates by ID
    collection.upsert(
        ids=ids,
        metadatas=metadatas,
        documents=documents,
        embeddings=embeddings
    )
    client.persist()
    return collection

# 4. Search knowledge base (vector DB) based on user input
def search_vector_db(
    query: str,
    collection: chromadb.api.models.Collection.Collection,
    embedder: OpenAIEmbeddings,
    k: int = 5,
    filter_meta: Optional[Dict] = None
) -> List[Dict]:
    """
    Returns top-k results with scores and metadata.
    filter_meta: optional metadata filter dict (exact match)
    """
    q_emb = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=filter_meta or {}
    )
    # results contains ids, documents, metadatas, distances
    hits = []
    for idx in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][idx],
            "text": results["documents"][0][idx],
            "meta": results["metadatas"][0][idx],
            "distance": results["distances"][0][idx]
        })
    return hits

# 5. Retrieve knowledge base result
def retrieve_kb_result(hits: List[Dict]) -> str:
    """
    Simple synthesizer that concatenates top hits into a single context string.
    You can replace with a smarter reranker or summarizer.
    """
    # Concatenate with separators and return
    parts = []
    for i, h in enumerate(hits):
        parts.append(f"Result {i+1} (id={h['id']})\n{h['text']}")
    return "\n\n---\n\n".join(parts)

# 6. Accuracy of response
def evaluate_retrieval_accuracy(
    retrieved_hits: List[Dict],
    relevant_doc_ids: List[str],
    k: int = 5
) -> Dict:
    """
    Compute recall@k — fraction of relevant_doc_ids found in top-k retrieved.
    """
    found = 0
    retrieved_ids = [h["meta"].get("doc_id") for h in retrieved_hits]
    for rid in relevant_doc_ids:
        if rid in retrieved_ids[:k]:
            found += 1
    recall_at_k = found / max(1, len(relevant_doc_ids))
    return {"recall_at_k": recall_at_k, "retrieved_ids": retrieved_ids[:k]}

def evaluate_response_accuracy(
    response_text: str,
    gold_text: str,
    embedder: OpenAIEmbeddings
) -> Dict:
    """
    Two signals:
      1) semantic similarity via embeddings (cosine)
      2) token-level F1 (approximate)
    Returns dict with both scores.
    """
    # Embedding similarity
    emb_resp = embedder.embed_query(response_text)
    emb_gold = embedder.embed_query(gold_text)
    cos_sim = float(cosine_similarity([emb_resp], [emb_gold])[0][0])

    # Token F1
    token_f1 = f1_score_tokens(response_text, gold_text)

    return {"cosine_similarity": cos_sim, "token_f1": token_f1}

# 7. Main orchestration function
def main_pipeline(
    docs: List[Dict],
    user_queries: List[Dict],
    collection_name: str = "kb_collection",
    persist_directory: str = "./chromadb_store",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embed_model: str = "text-embedding-3-small",
    top_k: int = 5,
    openai_api_key: Optional[str] = None
) -> Dict:
    """
    docs: list of {'id':..., 'text':..., 'meta':{...}}
    user_queries: list of {'query':..., 'gold_doc_ids':[...], 'gold_answer': optional str}
    Returns: dict with stored collection info and per-query metrics.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # 1. Chunk
    chunks = chunk_texts(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_meta=True)

    # 2. Embeddings for chunks
    texts = [c["text"] for c in chunks]
    embedder = OpenAIEmbeddings(model=embed_model)
    embeddings = create_embeddings(texts, model=embed_model)

    # 3. Store in Chroma
    collection = store_in_chroma(collection_name, chunks, embeddings, persist_directory=persist_directory)

    results = {"collection_name": collection_name, "num_chunks": len(chunks), "queries": []}

    # For each user query: search, retrieve, evaluate
    for q in user_queries:
        query_text = q["query"]
        gold_doc_ids = q.get("gold_doc_ids", [])
        gold_answer = q.get("gold_answer")  # optional reference answer for response evaluation

        hits = search_vector_db(query_text, collection, embedder, k=top_k)
        context = retrieve_kb_result(hits)

        # If you want to generate an LLM response using retrieved context, call your LLM here.
        # Example: response = llm(f"Use context:\n{context}\nAnswer: {query_text}")
        # For this code we will treat the concatenated context as the "response" when gold_answer is None.
        response_text = q.get("generated_response") or context

        retrieval_metrics = evaluate_retrieval_accuracy(hits, gold_doc_ids, k=top_k)
        response_metrics = {}
        if gold_answer:
            response_metrics = evaluate_response_accuracy(response_text, gold_answer, embedder)

        results["queries"].append({
            "query": query_text,
            "hits": [{"id": h["id"], "doc_id": h["meta"].get("doc_id"), "distance": h["distance"]} for h in hits],
            "retrieval_metrics": retrieval_metrics,
            "response_metrics": response_metrics,
            "response_text_snippet": response_text[:1000]
        })

    return results

# Example minimal usage
if __name__ == "__main__":
    # Load API key via env or set it here (not recommended to hardcode)
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)

    # Sample docs
    docs = [
        {"id": "doc1", "text": "LangChain is a library for building applications with LLMs. It provides chains, agents, and integrations.", "meta": {"title": "LangChain intro"}},
        {"id": "doc2", "text": "Chroma is an open-source embedding database designed for local and cloud usage. It supports simple Python API.", "meta": {"title": "Chroma intro"}},
    ]

    # Sample queries with ground truth doc ids and optional gold answer
    user_queries = [
        {"query": "What is LangChain used for?", "gold_doc_ids": ["doc1"], "gold_answer": "LangChain is a library for building applications with LLMs, providing chains and integrations."},
        {"query": "What is Chroma?", "gold_doc_ids": ["doc2"], "gold_answer": "Chroma is an open-source embedding database with a Python API for local and cloud use."}
    ]

    out = main_pipeline(docs, user_queries, openai_api_key=OPENAI_KEY)
    print(json.dumps(out, indent=2))
