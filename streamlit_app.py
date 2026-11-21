import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="CURAJ Smart Search (Hybrid RAG)", layout="wide")

# 1. Knowledge Base (Official CURAJ Ordinance 03 Rules & M.Sc. Courses)
documents = [
    "Attendance Policy (CURAJ Ordinance 03, Clause 6.1): Students must maintain a minimum of 75% attendance in registered courses to sit for lab and End of Semester Exams (EoSE).",
    "Attendance Condonation Rule (CURAJ Ordinance 03, Clause 6.6): In exceptional medical or unavoidable cases, the Vice-Chancellor may condone attendance shortage down to 65%.",
    "Python Lab Rule: Students must submit weekly Python practice programs and clear the lab viva.",
    "Course 6.0MBD01 (Linear Algebra): Covers vector spaces, eigenvalues, orthogonality, and MATLAB.",
    "Course 6.0MBD02 (Foundations of Data Science): Covers data preprocessing, PCA, SVD, LDA, and UMAP.",
    "Course 6.0MBD03 (DBMS): Covers ER diagrams, relational algebra, normalization, and MySQL.",
    "Course 6.0MBD71 (Programming in Python): Covers variables, data types, conditionals, loops, strings, lists, dictionaries, and functions.",
]


# 2. Beginner Helper Functions (Only Loops, Strings, Lists & If/Else)
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


# 3. Load Models Once (Keeps the Web App Fast)
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

# 4. Clear, Self-Explanatory Header
st.title("CURAJ Smart Policy & Course Search")
st.markdown(
    "Ask a question about **CURAJ attendance rules** or **M.Sc. Big Data courses**. "
    "This demo shows how **Hybrid RAG** (Meaning + Exact Keywords + AI Reranking) prevents standard AI search mistakes."
)

# 5. Search Controls
col_pick, col_custom = st.columns(2)

with col_pick:
    demo_query = st.selectbox(
        "Try a preset test question:",
        [
            "What topics are covered in 6.0MBD71?",
            "How much class presence is needed so I don't get detained?",
            "What does Clause 6.6 say about medical attendance shortage?",
        ],
    )

with col_custom:
    custom_query = st.text_input(
        "Or type your own question and press Enter:",
        placeholder="e.g., Which subject teaches MySQL? or Do I need lab viva?",
    )

if custom_query.strip() != "":
    query = custom_query
else:
    query = demo_query

# 6. Stage 1: Naive Vector Search (Meaning Only)
query_vec = embed_model.encode(query, normalize_embeddings=True)
vector_results = []

for i in range(len(documents)):
    score = cosine_similarity(query_vec, doc_vectors[i])
    vector_results.append((i, score))

vector_results.sort(key=get_score, reverse=True)

# 7. Stage 2: BM25 Keyword Search + Reciprocal Rank Fusion (RRF)
query_tokens = tokenize(query)
bm25_raw = bm25.get_scores(query_tokens)
bm25_results = []

for i in range(len(documents)):
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

# 8. Stage 3: Cross-Encoder Reranker on Top 3 Hybrid Matches
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

# 9. SHOW THE PRACTICAL ANSWER FIRST (Instant Clarity for Any Visitor)
best_doc_id = rerank_results[0][0]
best_score = rerank_results[0][1]
vector_top_id = vector_results[0][0]

st.divider()
st.subheader("Best Verified Answer (Context Sent to LLM)")

if best_score < -4.0:
    st.error(
        "No strong match found in the CURAJ database for this question. "
        "Open the Knowledge Base below to see what topics are indexed."
    )
    st.write(f"Closest weak guess ({best_score:+.2f}): {documents[best_doc_id]}")
else:
    st.success(documents[best_doc_id])

    if best_doc_id != vector_top_id:
        st.warning(
            "**Why Hybrid RAG mattered here:** Standard AI (Vector Search alone) missed this at #1 "
            "because exact course codes or clause numbers confuse meaning-only search. "
            "Adding BM25 keyword matching + AI Reranking caught the exact match and moved it to #1."
        )
    else:
        st.info(
            "**Why AI Search worked here:** Even if your wording didn't match the exact policy words, "
            "Vector Search understood the meaning at #1—and the Hybrid + Reranker pipeline verified it."
        )

# 10. UNDER THE HOOD: SIDE-BY-SIDE 3-STAGE COMPARISON
st.subheader("Under the Hood: How the 3-Stage Engine Ranked Results")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. Standard AI (Vector)")
    st.caption("Matches meaning only. Often misses exact codes.")
    for i in range(3):
        doc_id = vector_results[i][0]
        score = vector_results[i][1]
        st.write(f"**#{i + 1} (`{score:.4f}`)** — {documents[doc_id]}")

with col2:
    st.markdown("### 2. Hybrid (Vector + BM25)")
    st.caption("Combines meaning + exact keywords using RRF.")
    for i in range(3):
        doc_id = hybrid_results[i][0]
        score = hybrid_results[i][1]
        st.write(f"**#{i + 1} (`{score:.4f}`)** — {documents[doc_id]}")

with col3:
    st.markdown("### 3. AI Reranker (Final)")
    st.caption("Reads question + Top 3 together to pick the winner.")
    for i in range(len(rerank_results)):
        doc_id = rerank_results[i][0]
        score = rerank_results[i][1]
        st.write(f"**#{i + 1} (`{score:+.4f}`)** — {documents[doc_id]}")

# 11. TRANSPARENCY PANEL (So Any Visitor Knows What the App Searches)
st.divider()
with st.expander("View Knowledge Base & Official Source (What this app searches)"):
    for i in range(len(documents)):
        st.write(f"{i + 1}. {documents[i]}")
    st.caption(
        "Official Attendance Policy Source: Central University of Rajasthan, "
        "Ordinance 03 (Examination and Evaluation), Clauses 6.1 & 6.6 — "
        "https://www.curaj.ac.in/sites/default/files/University%20Ordinance%20Revised%20No.%2003.pdf"
    )
    st.caption(
        "Code Built With: Python Lists, Dictionaries, for Loops, if/else & Functions "
        "(Course 6.0MBD71 — No NumPy / No OOP)"
    )
