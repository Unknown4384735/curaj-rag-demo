import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

documents = [
    "Attendance Policy: Students need at least 75% attendance to sit for lab and end-term exams.",
    "Python Lab Rule: Students must submit weekly Python practice programs and clear the lab viva.",
    "Course 6.0MBD01 (Linear Algebra): Covers vector spaces, eigenvalues, orthogonality, and MATLAB.",
    "Course 6.0MBD02 (Foundations of Data Science): Covers data preprocessing, PCA, SVD, LDA, and UMAP.",
    "Course 6.0MBD03 (DBMS): Covers ER diagrams, relational algebra, normalization, and MySQL.",
    "Course 6.0MBD71 (Programming in Python): Covers variables, data types, conditionals, loops, strings, lists, dictionaries, and functions.",
]

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


# 3. Load Models 
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    doc_vectors = embed_model.encode(documents, normalize_embeddings=True)

    tokenized_docs = []
    for doc in documents:
        tokenized_docs.append(tokenize(doc))
    bm25 = BM25Okapi(tokenized_docs)

    return embed_model, rerank_model, doc_vectors, bm25


embed_model, rerank_model, doc_vectors, bm25 = load_models()

# 4. Simple Web Page Layout
st.title("CURAJ Hybrid RAG Search Engine")
st.caption("Programming in Python (6.0MBD71) — Lists, Dictionaries, Loops, If/Else & Functions")

demo_query = st.selectbox(
    "Choose a presentation demo query:",
    [
        "What topics are covered in 6.0MBD71?",
        "How much class presence is needed so I don't get detained?",
    ],
)

custom_query = st.text_input("Or type your own question and press Enter:")

if custom_query.strip() != "":
    query = custom_query
else:
    query = demo_query

# 5. Stage 1: Naive Vector Search (Meaning Only)
query_vec = embed_model.encode(query, normalize_embeddings=True)
vector_results = []

for i in range(len(documents)):
    score = cosine_similarity(query_vec, doc_vectors[i])
    vector_results.append((i, score))

vector_results.sort(key=get_score, reverse=True)

# 6. Stage 2: BM25 Keyword Search + Reciprocal Rank Fusion (RRF)
query_tokens = tokenize(query)
bm25_raw = bm25.get_scores(query_tokens)
bm25_results = []

for i in range(len(documents)):
    score = float(bm25_raw[i])
    if score > 0:
        bm25_results.append((i, score))

bm25_results.sort(key=get_score, reverse=True)

# Combine Vector Ranks and BM25 Ranks using a Python Dictionary
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

# 7. Stage 3: Cross-Encoder Reranker on Top 3 Hybrid Matches
top_3 = hybrid_results[:3]
pairs = []

for item in top_3:
    doc_id = item[0]
    pairs.append([query, documents[doc_id]])

rerank_raw = rerank_model.predict(pairs)
rerank_results = []

for i in range(len(top_3)):
    doc_id = top_3[i][0]
    rerank_results.append((doc_id, float(rerank_raw[i])))

rerank_results.sort(key=get_score, reverse=True)

# 8. All 3 Stages Clearly on Screen
st.subheader("1) Naive Vector Search (Top 3)")
for i in range(3):
    doc_id = vector_results[i][0]
    score = vector_results[i][1]
    st.write(f"**#{i + 1} ({score:.4f})** → {documents[doc_id]}")

st.subheader("2) Hybrid Search: Vector + BM25 + RRF (Top 3)")
for i in range(3):
    doc_id = hybrid_results[i][0]
    score = hybrid_results[i][1]
    st.write(f"**#{i + 1} ({score:.4f})** → {documents[doc_id]}")

st.subheader("3) After Cross-Encoder Reranker (Final Ranking)")
for i in range(len(rerank_results)):
    doc_id = rerank_results[i][0]
    score = rerank_results[i][1]
    st.write(f"**#{i + 1} ({score:+.4f})** → {documents[doc_id]}")

st.divider()
st.subheader("Final Context Sent to LLM")
best_doc_id = rerank_results[0][0]
st.success(documents[best_doc_id])
