# curaj-rag-demo

# Hybrid RAG Search Engine (6.0MBD71)

A beginner-friendly Retrieval-Augmented Generation (RAG) search engine built for **Programming in Python (6.0MBD71)** using core Python fundamentals.

## What It Demonstrates
This app compares three retrieval stages on a small university policy and course syllabus knowledge base:

1. **Naive Vector Search (Semantic Meaning):** Uses `all-MiniLM-L6-v2` embeddings and a manual `for`-loop Cosine Similarity function. Great for meaning-based queries (e.g., matching *"class presence"* to *"attendance"*), but can miss exact course codes.
2. **Hybrid Search (Vectors + BM25 + RRF):** Combines semantic rankings with BM25 exact keyword matching using Reciprocal Rank Fusion (`1 / (60 + rank)`).
3. **Cross-Encoder Reranking:** Re-scores the top 3 hybrid candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the final context sent to an LLM.

## Demo Queries Included
- **Meaning Query:** *"How much class presence is needed so I don't get detained?"* — Vector search ranks the 75% Attendance Policy at `#1`.
- **Exact Code Query:** *"What topics are covered in 6.0MBD71?"* — Naive vector search misses `6.0MBD71` in the Top 3, while Hybrid Search + Cross-Encoder Reranking lifts it to `#1`.

## Python Concepts Used
Built strictly with foundational Python (no NumPy, no OOP/classes):
- **Lists & Tuples** — storing documents, tokens, and `(doc_id, score)` pairs
- **Dictionaries** — accumulating Reciprocal Rank Fusion (`rrf_scores`)
- **`for` Loops & `if/else` Conditions** — tokenization, dot-product similarity, and score filtering
- **Functions** — modular helpers (`tokenize`, `cosine_similarity`, `get_score`)

## Tech Stack
- **UI:** Streamlit
- **Embeddings & Reranking:** `sentence-transformers` (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`)
- **Keyword Search:** `rank-bm25`

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
