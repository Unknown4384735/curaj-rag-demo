import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="CURAJ Hybrid RAG PDF Search", layout="wide")

CHUNK_SIZE, STEP_SIZE, CUTOFF = 140, 100, -5.0
STOP_WORDS = [
    "a", "an", "the", "is", "are", "was", "were", "be", "in", "on", "at",
    "to", "for", "of", "and", "or", "by", "as", "with", "from", "about",
    "what", "which", "who", "how", "when", "where", "why", "do", "does",
    "did", "can", "i", "we", "you", "my", "our", "it", "this", "that", "so"
]


# 1. Clean text & drop filler words (keeps dots for codes like 6.0MBD71)
def tokenize(text):
    clean = ""
    for ch in text.lower():
        if ch.isalnum() or ch == ".":
            clean += ch
        else:
            clean += " "

    words = clean.split()
    tokens = []
    for w in words:
        w = w.strip(".")
        if w != "" and w not in STOP_WORDS:
            tokens.append(w)
    return tokens if tokens else words


# 2. Cosine Similarity (plain loop, no NumPy — vectors are already length 1)
def cosine_sim(v1, v2):
    total = 0.0
    for i in range(len(v1)):
        total += float(v1[i]) * float(v2[i])
    return total


# 3. Split uploaded PDF into small overlapping chunks
def chunk_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    chunks = []
    for page_num, page in enumerate(reader.pages):
        words = (page.extract_text() or "").split()
        part = 1
        for start in range(0, len(words), STEP_SIZE):
            chunks.append({
                "title": f"{pdf_file.name} — Page {page_num + 1} (Part {part})",
                "text": " ".join(words[start : start + CHUNK_SIZE]),
            })
            part += 1
            if start + CHUNK_SIZE >= len(words):
                break
    return chunks


# 4. Cache AI models & PDF index so searches stay instant
@st.cache_resource
def load_models():
    return (
        SentenceTransformer("all-MiniLM-L6-v2"),
        CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )


@st.cache_resource
def build_index(texts, _embed_model):
    vecs = _embed_model.encode(texts, normalize_embeddings=True)
    tokenized = []
    for t in texts:
        tokenized.append(tokenize(t))
    return vecs, BM25Okapi(tokenized)


def show_top3(col, title, results, docs):
    with col:
        st.markdown(f"### {title}")
        for i in range(min(3, len(results))):
            doc_id, score = results[i]
            st.write(f"**#{i + 1} (`{score:.4f}`)** — {docs[doc_id]['title']}")
            st.caption(docs[doc_id]["text"][:110] + "...")


# --- APP UI ---
st.title("Hybrid RAG PDF Search Engine")

with st.spinner("Loading AI models..."):
    embed_model, rerank_model = load_models()

files = st.file_uploader("Upload PDF file(s):", type="pdf", accept_multiple_files=True)
if not files:
    st.info("Upload a text-based PDF above to begin.")
    st.stop()

docs = []
for f in files:
    docs.extend(chunk_pdf(f))

if not docs:
    st.warning("No readable text found in the uploaded PDF.")
    st.stop()

texts = []
for d in docs:
    texts.append(d["text"])

with st.spinner("Indexing PDF..."):
    doc_vecs, bm25 = build_index(texts, embed_model)

st.caption(f"Ready: Indexed {len(docs)} chunks.")
query = st.text_input("Ask a question:", placeholder="e.g., What topics are covered in 6.0MBD71?")
if not query.strip():
    st.stop()


# --- STAGE 1: VECTOR SEARCH (Meaning) ---
q_vec = embed_model.encode(query, normalize_embeddings=True)
vec_res = []
for i in range(len(texts)):
    vec_res.append((i, cosine_sim(q_vec, doc_vecs[i])))
vec_res.sort(key=lambda x: x[1], reverse=True)


# --- STAGE 2: BM25 (Keywords) + RRF MERGER (Dictionary) ---
bm25_raw = bm25.get_scores(tokenize(query))
bm25_res = []
for i in range(len(texts)):
    if bm25_raw[i] > 0:
        bm25_res.append((i, float(bm25_raw[i])))
bm25_res.sort(key=lambda x: x[1], reverse=True)

rrf = {}
for rank in range(len(vec_res)):
    doc_id = vec_res[rank][0]
    rrf[doc_id] = 1.0 / (60 + rank + 1)
for rank in range(len(bm25_res)):
    doc_id = bm25_res[rank][0]
    rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (60 + rank + 1)

hybrid_res = list(rrf.items())
hybrid_res.sort(key=lambda x: x[1], reverse=True)


# --- STAGE 3: AI RERANKER (Top 10 Side-by-Side) ---
top10 = hybrid_res[:10]
pairs = []
for doc_id, _ in top10:
    pairs.append([query, texts[doc_id]])

rerank_raw = rerank_model.predict(pairs)
rerank_res = []
for i in range(len(top10)):
    rerank_res.append((top10[i][0], float(rerank_raw[i])))
rerank_res.sort(key=lambda x: x[1], reverse=True)


# --- SHOW RESULT + GUARDRAIL ---
best_id, best_score = rerank_res[0]
st.divider()
st.subheader("Best Verified Match")

if best_score < CUTOFF:
    st.error("No strong match found in the uploaded PDF(s).")
else:
    st.success(f"**{docs[best_id]['title']}**\n\n{docs[best_id]['text']}")
    if best_id != vec_res[0][0]:
        st.warning("**Hybrid RAG rescued this answer:** Vector Search missed it at #1, while BM25 + RRF + Reranking found it.")
    else:
        st.info("**Verified match:** Vector Search and Hybrid + Reranking both agreed on #1.")

col1, col2, col3 = st.columns(3)
show_top3(col1, "1. Vector Search", vec_res, docs)
show_top3(col2, "2. Hybrid (Vector + BM25)", hybrid_res, docs)
show_top3(col3, "3. AI Reranker (Final)", rerank_res, docs)
