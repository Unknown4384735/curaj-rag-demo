import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="CURAJ Hybrid RAG PDF Search", layout="wide")


# 1. Beginner Helper Functions (Lists, Dictionaries, Loops, If/Else Only)
def tokenize(text):
    cleaned = ""
    for ch in text.lower():
        if ch.isalnum() or ch == ".":
            cleaned = cleaned + ch
        else:
            cleaned = cleaned + " "

    words = cleaned.split()
    tokens = []
    for word in words:
        tokens.append(word.strip("."))
    return tokens


def cosine_similarity(vec1, vec2):
    total = 0.0
    for i in range(len(vec1)):
        total = total + float(vec1[i]) * float(vec2[i])
    return total


def get_score(item):
    return item[1]


def chunk_pdf(pdf_source, file_name):
    reader = PdfReader(pdf_source)
    chunks = []
    chunk_size = 140  # words per chunk
    step = 100        # 40-word overlap so codes/sentences aren't cut in half

    for page_num in range(len(reader.pages)):
        raw_text = reader.pages[page_num].extract_text()
        if raw_text is None or raw_text.strip() == "":
            continue

        words = raw_text.split()
        part = 1
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + chunk_size])
            chunks.append(
                {
                    "title": f"{file_name} — Page {page_num + 1} (Part {part})",
                    "text": piece,
                }
            )
            part = part + 1
            if start + chunk_size >= len(words):
                break

    return chunks


# 2. Load AI Models Once + Cache Index Per PDF Set
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embed_model, rerank_model


@st.cache_resource
def build_index(text_list, _embed_model):
    doc_vectors = _embed_model.encode(text_list, normalize_embeddings=True)

    tokenized_docs = []
    for text in text_list:
        tokenized_docs.append(tokenize(text))
    bm25 = BM25Okapi(tokenized_docs)

    return doc_vectors, bm25


embed_model, rerank_model = load_models()

# 3. Header & PDF Uploader
st.title("Hybrid RAG PDF Search Engine")
st.markdown(
    "Upload any PDF (official university ordinance, course syllabus, or notes) and ask a question. "
    "Extracts **100% raw PDF text** and compares Standard Vector Search vs. Hybrid Search (BM25 + RRF) + Reranking."
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files:",
    type="pdf",
    accept_multiple_files=True,
)

# 4. Read Uploaded PDFs (or Auto-Load Any PDFs Placed in the Repo Folder)
documents = []

if len(uploaded_files) > 0:
    for pdf_file in uploaded_files:
        pdf_chunks = chunk_pdf(pdf_file, pdf_file.name)
        for chunk in pdf_chunks:
            documents.append(chunk)
else:
    repo_files = sorted(os.listdir("."))
    for filename in repo_files:
        if filename.lower().endswith(".pdf"):
            pdf_chunks = chunk_pdf(filename, filename)
            for chunk in pdf_chunks:
                documents.append(chunk)

if len(documents) == 0:
    st.info(
        "Upload a text-based PDF above (for example, your CURAJ Ordinance 03 or M.Sc. Syllabus PDF) to begin."
    )
    st.stop()

search_texts = []
for doc in documents:
    search_texts.append(doc["text"])

with st.spinner("Indexing PDF pages..."):
    doc_vectors, bm25 = build_index(search_texts, embed_model)

st.caption(f"Ready: Indexed {len(documents)} raw text chunks directly from PDF.")

# 5. Search Box
query = st.text_input(
    "Ask a question and press Enter:",
    placeholder="e.g., What topics are covered in 6.0MBD71? or minimum attendance required",
)

if query.strip() == "":
    st.stop()

# 6. Stage 1: Naive Vector Search
query_vec = embed_model.encode(query, normalize_embeddings=True)
vector_results = []

for i in range(len(search_texts)):
    score = cosine_similarity(query_vec, doc_vectors[i])
    vector_results.append((i, score))

vector_results.sort(key=get_score, reverse=True)

# 7. Stage 2: BM25 Keyword Search + Reciprocal Rank Fusion (RRF)
query_tokens = tokenize(query)
bm25_raw = bm25.get_scores(query_tokens)
bm25_results = []

for i in range(len(search_texts)):
    score = float(bm25_raw[i])
    if score > 0:
        bm25_results.append((i, score))

bm25_results.sort(key=get_score, reverse=True)

rrf_scores = {}

for i in range(len(vector_results)):
    doc_id = vector_results[i][0]
    rank = i + 1
    rrf_scores[doc_id] = 1.0 / (60 + rank)

for i in range(len(bm25_results)):
    doc_id = bm25_results[i][0]
    rank = i + 1
    if doc_id in rrf_scores:
        rrf_scores[doc_id] = rrf_scores[doc_id] + (1.0 / (60 + rank))
    else:
        rrf_scores[doc_id] = 1.0 / (60 + rank)

hybrid_results = []
for doc_id in rrf_scores:
    hybrid_results.append((doc_id, rrf_scores[doc_id]))

hybrid_results.sort(key=get_score, reverse=True)

# 8. Stage 3: Cross-Encoder Reranker on Top 5 Hybrid Matches
top_k = min(5, len(hybrid_results))
top_candidates = hybrid_results[:top_k]
pairs = []

for item in top_candidates:
    doc_id = item[0]
    pairs.append([query, search_texts[doc_id]])

rerank_raw = rerank_model.predict(pairs)
rerank_results = []

for i in range(len(top_candidates)):
    doc_id = top_candidates[i][0]
    rerank_results.append((doc_id, float(rerank_raw[i])))

rerank_results.sort(key=get_score, reverse=True)

# 9. Show Best Match First (Exact Raw PDF Text + Page Number)
best_doc_id = rerank_results[0][0]
best_score = rerank_results[0][1]
vector_top_id = vector_results[0][0]

st.divider()
st.subheader("Best Verified Match (Raw PDF Excerpt Sent to LLM)")

if best_score < -4.0:
    st.error("No strong match found in the uploaded PDF(s) for this question.")
else:
    st.success(f"**{documents[best_doc_id]['title']}**\n\n{documents[best_doc_id]['text']}")

    if best_doc_id != vector_top_id:
        st.warning(
            "**Hybrid RAG rescued this answer:** Standard Vector Search missed this page at #1, "
            "while BM25 + RRF + Cross-Encoder Reranking found the exact match."
        )
    else:
        st.info(
            "**Verified match:** Vector Search ranked this page #1 by meaning, and Hybrid + Reranking confirmed it."
        )

# 10. Clean 3-Column Comparison (Page Citations + Short Preview)
st.subheader("Under the Hood: 3-Stage Ranking Comparison")
col1, col2, col3 = st.columns(3)
show_count = min(3, len(documents))

with col1:
    st.markdown("### 1. Vector Search")
    for i in range(show_count):
        doc_id = vector_results[i][0]
        score = vector_results[i][1]
        preview = documents[doc_id]["text"][:110] + "..."
        st.write(f"**#{i + 1} (`{score:.4f}`)** — {documents[doc_id]['title']}")
        st.caption(preview)

with col2:
    st.markdown("### 2. Hybrid (Vector + BM25)")
    for i in range(show_count):
        doc_id = hybrid_results[i][0]
        score = hybrid_results[i][1]
        preview = documents[doc_id]["text"][:110] + "..."
        st.write(f"**#{i + 1} (`{score:.4f}`)** — {documents[doc_id]['title']}")
        st.caption(preview)

with col3:
    st.markdown("### 3. AI Reranker (Final)")
    for i in range(min(3, len(rerank_results))):
        doc_id = rerank_results[i][0]
        score = rerank_results[i][1]
        preview = documents[doc_id]["text"][:110] + "..."
        st.write(f"**#{i + 1} (`{score:+.4f}`)** — {documents[doc_id]['title']}")
        st.caption(preview)
