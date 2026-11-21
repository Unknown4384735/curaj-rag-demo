import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="CURAJ Smart Search (Hybrid RAG)", layout="wide")

# 1. Knowledge Base (List of Dictionaries — paste raw PDF text or type your own words in "text")
documents = [
    {
        "title": "CURAJ Ordinance 03 — Clause 6.1 (Attendance)",
        "text": "PASTE_RAW_PDF_TEXT_OR_TYPE_YOUR_OWN_WORDS_HERE",
    },
    {
        "title": "CURAJ Ordinance 03 — Clause 6.6 (Medical Condonation)",
        "text": "PASTE_RAW_PDF_TEXT_OR_TYPE_YOUR_OWN_WORDS_HERE",
    },
    {
        "title": "6.0MBD01 — Database Management Systems",
        "text": "PASTE_RAW_PDF_TEXT_OR_TYPE_YOUR_OWN_WORDS_HERE",
    },
    {
        "title": "6.0MBD02 — Foundations of Data Science",
        "text": "PASTE_RAW_PDF_TEXT_OR_TYPE_YOUR_OWN_WORDS_HERE",
    },
    {
        "title": "6.0MBD03 — Fundamentals of Probability and Statistics",
        "text": "PASTE_RAW_PDF_TEXT_OR_TYPE_YOUR_OWN_WORDS_HERE",
    },
    {
        "title": "6.0MBD04 — Linear Algebra and its Applications",
        "text": "PASTE_RAW_PDF_TEXT_OR_TYPE_YOUR_OWN_WORDS_HERE",
    },
    {
        "title": "6.0MBD71 — Programming in Python",
        "text": "PASTE_RAW_PDF_TEXT_OR_TYPE_YOUR_OWN_WORDS_HERE",
    },
    {
        "title": "6.0MBD81 — Professional Communication",
        "text": "PASTE_RAW_PDF_TEXT_OR_TYPE_YOUR_OWN_WORDS_HERE",
    },
]

# Combine title + text into plain strings for searching
search_texts = []
for doc in documents:
    search_texts.append(doc["title"] + "\n" + doc["text"])


# 2. Beginner Helper Functions
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


# 3. Load Models Once (Auto-refreshes whenever search_texts changes)
@st.cache_resource
def load_models(text_list):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    doc_vectors = embed_model.encode(text_list, normalize_embeddings=True)

    tokenized_docs = []
    for text in text_list:
        tokenized_docs.append(tokenize(text))
    bm25 = BM25Okapi(tokenized_docs)

    return embed_model, rerank_model, doc_vectors, bm25


embed_model, rerank_model, doc_vectors, bm25 = load_models(search_texts)

# 4. Header & Search Controls
st.title("CURAJ Smart Policy & Course Search")

col_pick, col_custom = st.columns(2)
with col_pick:
    demo_query = st.selectbox(
        "Preset questions:",
        [
            "What topics are covered in 6.0MBD71?",
            "How much class presence is needed so I don't get detained?",
            "What does Clause 6.6 say about medical attendance shortage?",
        ],
    )

with col_custom:
    custom_query = st.text_input("Or type your own question and press Enter:")

if custom_query.strip() != "":
    query = custom_query
else:
    query = demo_query

# 5. Stage 1: Vector Search
query_vec = embed_model.encode(query, normalize_embeddings=True)
vector_results = []

for i in range(len(search_texts)):
    score = cosine_similarity(query_vec, doc_vectors[i])
    vector_results.append((i, score))

vector_results.sort(key=get_score, reverse=True)

# 6. Stage 2: BM25 + RRF
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

# 7. Stage 3: Cross-Encoder Reranker
top_3 = hybrid_results[:3]
pairs = []

for item in top_3:
    doc_id = item[0]
    pairs.append([query, search_texts[doc_id]])

rerank_raw = rerank_model.predict(pairs)
rerank_results = []

for i in range(len(top_3)):
    doc_id = top_3[i][0]
    rerank_results.append((doc_id, float(rerank_raw[i])))

rerank_results.sort(key=get_score, reverse=True)

# 8. Top Box: Show Full Text ONLY for the Single Winning Match
best_doc_id = rerank_results[0][0]
best_score = rerank_results[0][1]

st.divider()
st.subheader("Top Match (Context Sent to LLM)")

if best_score < -4.0:
    st.error("No strong match found in the database.")
else:
    st.success(f"**{documents[best_doc_id]['title']}**\n\n{documents[best_doc_id]['text']}")

# 9. Clean 3-Column Comparison (Short Titles Only — Never Messy)
st.subheader("3-Stage Ranking Comparison")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. Vector Search")
    for i in range(3):
        doc_id = vector_results[i][0]
        score = vector_results[i][1]
        st.write(f"**#{i + 1} (`{score:.4f}`)** — {documents[doc_id]['title']}")

with col2:
    st.markdown("### 2. Hybrid (Vector + BM25)")
    for i in range(3):
        doc_id = hybrid_results[i][0]
        score = hybrid_results[i][1]
        st.write(f"**#{i + 1} (`{score:.4f}`)** — {documents[doc_id]['title']}")

with col3:
    st.markdown("### 3. Reranker (Final)")
    for i in range(len(rerank_results)):
        doc_id = rerank_results[i][0]
        score = rerank_results[i][1]
        st.write(f"**#{i + 1} (`{score:+.4f}`)** — {documents[doc_id]['title']}")

# 10. Full Database Viewer (Hidden Until Clicked)
st.divider()
with st.expander("View All Indexed Documents"):
    for doc in documents:
        st.markdown(f"**{doc['title']}**")
        st.write(doc["text"])
        st.markdown("---")
