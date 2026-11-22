import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="CURAJ Hybrid RAG PDF Search", layout="wide")

# ==========================================================
# SETTINGS (Easy to tune in one place)
# ==========================================================
CHUNK_SIZE = 140        # Smaller chunks keep each course separate from old tables
STEP_SIZE = 100         # 40-word overlap so codes/sentences aren't cut in half
GUARDRAIL_CUTOFF = -5.0 # Below this Reranker score = off-topic question


# ==========================================================
# HELPER FUNCTIONS (100% Core Python: Loops, Lists, Strings)
# ==========================================================

# 1. Clean text and remove common filler words for BM25 keyword search
def tokenize(text):
    stop_words = [
        "a", "an", "the", "is", "are", "was", "were", "be", "in", "on", "at",
        "to", "for", "of", "and", "or", "by", "as", "with", "from", "about",
        "what", "which", "who", "how", "when", "where", "why", "do", "does",
        "did", "can", "i", "we", "you", "my", "our", "it", "this", "that", "so"
    ]

    # Keep only letters, numbers, and dots (dots keep codes like 6.0MBD71 intact)
    cleaned = ""
    for ch in text.lower():
        if ch.isalnum() or ch == ".":
            cleaned = cleaned + ch
        else:
            cleaned = cleaned + " "

    words = cleaned.split()
    tokens = []
    for word in words:
        clean_word = word.strip(".")
        if clean_word != "" and clean_word not in stop_words:
            tokens.append(clean_word)

    # Safety fallback if a query only had stop words
    if len(tokens) == 0:
        return words
    return tokens


# 2. Measure similarity between two meaning vectors (no NumPy needed)
# Note: Because vectors are normalized to length 1, Cosine Similarity = dot product
def cosine_similarity(vec1, vec2):
    total = 0.0
    for i in range(len(vec1)):
        total = total + float(vec1[i]) * float(vec2[i])
    return total


# 3. Helper so list.sort() knows to sort by the score (2nd item in each pair)
def get_score(item):
    return item[1]


# 4. Read every PDF page and split it into small overlapping chunks
def chunk_pdf(pdf_source, file_name):
    reader = PdfReader(pdf_source)
    chunks = []

    for page_num in range(len(reader.pages)):
        raw_text = reader.pages[page_num].extract_text()
        if raw_text is None or raw_text.strip() == "":
            continue

        words = raw_text.split()
        part = 1
        for start in range(0, len(words), STEP_SIZE):
            piece = " ".join(words[start : start + CHUNK_SIZE])
            chunks.append({
                "title": f"{file_name} — Page {page_num + 1} (Part {part})",
                "text": piece,
            })
            part = part + 1
            if start + CHUNK_SIZE >= len(words):
                break

    return chunks


# ==========================================================
# LOAD PRE-TRAINED AI MODELS ONCE (Cached for speed)
# ==========================================================
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embed_model, rerank_model


@st.cache_resource
def build_index(text_list, _embed_model):
    # Turn every PDF chunk into meaning vectors (Stage 1)
    doc_vectors = _embed_model.encode(text_list, normalize_embeddings=True)

    # Turn every PDF chunk into clean keyword lists for BM25 (Stage 2)
    tokenized_docs = []
    for text in text_list:
        tokenized_docs.append(tokenize(text))
    bm25 = BM25Okapi(tokenized_docs)

    return doc_vectors, bm25


embed_model, rerank_model = load_models()


# ==========================================================
# WEB PAGE UI (Header & PDF Loader)
# ==========================================================
st.title("Hybrid RAG PDF Search Engine")
st.markdown(
    "Ask a question about any university PDF. Compares **1. Vector Search (Meaning)** "
    "vs. **2. Hybrid Search (Meaning + Exact Keywords)** vs. **3. AI Reranker (Final Judge)**."
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files (or use the pre-loaded CURAJ syllabus):",
    type="pdf",
    accept_multiple_files=True,
)

# Load uploaded PDFs; if none uploaded, auto-load PDFs saved in this folder
documents = []
if len(uploaded_files) > 0:
    for pdf_file in uploaded_files:
        for chunk in chunk_pdf(pdf_file, pdf_file.name):
            documents.append(chunk)
else:
    for filename in sorted(os.listdir(".")):
        if filename.lower().endswith(".pdf"):
            for chunk in chunk_pdf(filename, filename):
                documents.append(chunk)

if len(documents) == 0:
    st.info("Upload a text-based PDF above to begin.")
    st.stop()

# Extract just the raw text from every chunk for searching
search_texts = []
for doc in documents:
    search_texts.append(doc["text"])

with st.spinner("Indexing PDF pages..."):
    doc_vectors, bm25 = build_index(search_texts, embed_model)

st.caption(f"Ready: Indexed {len(documents)} clean text chunks directly from PDF.")

# Search box
query = st.text_input(
    "Ask a question and press Enter:",
    placeholder="e.g., What topics are covered in 6.0MBD71? or minimum attendance required",
)

if query.strip() == "":
    st.stop()


# ==========================================================
# STAGE 1: VECTOR SEARCH (Meaning Only)
# ==========================================================
query_vec = embed_model.encode(query, normalize_embeddings=True)
vector_results = []

for i in range(len(search_texts)):
    score = cosine_similarity(query_vec, doc_vectors[i])
    vector_results.append((i, score))

vector_results.sort(key=get_score, reverse=True)


# ==========================================================
# STAGE 2: BM25 KEYWORD SEARCH + RRF MERGER (Dictionary)
# ==========================================================
query_tokens = tokenize(query)
bm25_raw = bm25.get_scores(query_tokens)
bm25_results = []

for i in range(len(search_texts)):
    score = float(bm25_raw[i])
    if score > 0:
        bm25_results.append((i, score))

bm25_results.sort(key=get_score, reverse=True)

# Merge Vector ranks + BM25 ranks fairly using RRF: 1 / (60 + rank)
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


# ==========================================================
# STAGE 3: AI RERANKER (Reads Top 10 Side-by-Side)
# ==========================================================
top_k = min(10, len(hybrid_results))
top_candidates = hybrid_results[:top_k]

# Pair the user's question with each of the Top 10 chunks
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


# ==========================================================
# DISPLAY RESULTS + GUARDRAIL
# ==========================================================
best_doc_id = rerank_results[0][0]
best_score = rerank_results[0][1]
vector_top_id = vector_results[0][0]

st.divider()
st.subheader("Best Verified Match (Raw PDF Excerpt Sent to LLM)")

if best_score < GUARDRAIL_CUTOFF:
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

# 3-Column Comparison (With beginner-friendly captions)
st.subheader("Under the Hood: 3-Stage Ranking Comparison")
col1, col2, col3 = st.columns(3)
show_count = min(3, len(documents))

with col1:
    st.markdown("### 1. Vector Search")
    st.caption("Reads meaning only — misses exact course codes.")
    for i in range(show_count):
        doc_id = vector_results[i][0]
        score = vector_results[i][1]
        st.write(f"**#{i + 1} (`{score:.4f}`)** — {documents[doc_id]['title']}")
        st.caption(documents[doc_id]["text"][:110] + "...")

with col2:
    st.markdown("### 2. Hybrid (Vector + BM25)")
    st.caption("Combines meaning + exact keywords using RRF.")
    for i in range(show_count):
        doc_id = hybrid_results[i][0]
        score = hybrid_results[i][1]
        st.write(f"**#{i + 1} (`{score:.4f}`)** — {documents[doc_id]['title']}")
        st.caption(documents[doc_id]["text"][:110] + "...")

with col3:
    st.markdown("### 3. AI Reranker (Final)")
    st.caption("Reads question & top pages side-by-side to pick #1.")
    for i in range(min(3, len(rerank_results))):
        doc_id = rerank_results[i][0]
        score = rerank_results[i][1]
        st.write(f"**#{i + 1} (`{score:+.4f}`)** — {documents[doc_id]['title']}")
        st.caption(documents[doc_id]["text"][:110] + "...")
