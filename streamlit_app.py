import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="CURAJ Smart Search (Hybrid RAG)", layout="wide")

# 1. Knowledge Base (Official CURAJ Ordinance 03 Rules & M.Sc. Courses)
documents = [
    # Official CURAJ Ordinance 03 (Examination and Evaluation)
    "CURAJ Ordinance 03 — 6.1. A student shall be required to put in a minimum of 75% of attendance in the registered courses.",
    "CURAJ Ordinance 03 — 6.6. In very exceptional cases such as on medical or any other unavoidable grounds, the Vice-chancellor may condone shortage of attendance of a student under following conditions. a. Condonation cannot be for attendance less than 65% b. Condonation can be for only one occasion in the entire duration of the programme of Study.",
    "CURAJ Ordinance 03 — 13.2. To appear in the EoSE the student must have minimum attendance as prescribed in this ordinance.",

    # Official M.Sc. CS (BDA) Syllabus (July 2026 onwards) — Semester I Courses
    "Course: DATABASE MANAGEMENT SYSTEM | Code: 6.0MBD01 | Credits: 4 | Examination Scheme: Internal Assessment: 40 Marks (CIA-I: 20 Marks Unit I & II, CIA-II: 20 Marks Written Exams/ Quizzes/Assignment /Presentations/ Viva-Voce), End Semester Examination: 60 Marks (Unit I-IV). Unit-I: Basic Concepts: Need, purpose and goal of DBMS, Three-tier architecture, ER Diagram, data models- Relational, Network, Hierarchical. Database Design: Conceptual database design, concept of physical and logical databases, data abstraction and data independence, data aggregation. Unit-II: Relational Database: Relations, Relational Algebra, Theory of Normalization, Functional Dependency, Primitive and Composite data types. Unit-III: Application Development using SQL: DDL and DML, Host Language interface, embedded SQL programming, Stored procedures and triggers and views, Constraints assertions. NoSQL Databases-MongoDB. Unit-IV: Internal of RDBMS: Physical data organization in sequential, indexed random and hashed files. Inverted and multilist structures, B trees, B+ trees, Query Optimization, Join Algorithm, Statistics and Cost Base optimization. Parallel and distributed database. Transaction Processing, concurrency control, and recovery management. Transaction model properties and state Serializability. Lock base protocols, two-phase locking.",

    "Course: FOUNDATIONS OF DATA SCIENCE | Code: 6.0MBD02 | Credits: 4 | Examination Scheme: Internal Assessment: 40 Marks (CIA-I: 20 Marks Unit I & II, CIA-II: 20 Marks Written Exams/ Quizzes /Assignment /Presentations/ Viva-Voce), End Semester Examination: 60 Marks (Unit I-IV). Unit-I: Data Collection: Concepts of measurement, scales of measurement, design of data collection formats with illustration, data quality and issues with data collection systems with examples from business, Data preparation, Data pre-processing, Data Integration, Data Transformation. Unit-II: Graph: Introduction to Graph, Application of Graphs, Finite and Infinite Graphs, Incidence & Degree, Isolated Vertex, Pendant Vertex, Null Graph. Paths & Circuits: Isomorphism, Subgraphs, Walks, Paths, and Circuits, Connected Graphs, Disconnected Graphs, and Components, Euler Graphs, Hamiltonian Paths and Circuits. Trees & Fundamental Circuits: Spanning Trees. Cut-Sets & Cut-Vertices, Network Flows. Random Graphs: Large graphs, G(n,p) model, Giant Component, Connectivity, Cycles, Non-Uniform models, Applications. Unit-III: High Dimensional Space: Properties, Law of large number, Sphere and cube in high dimension, Generation points on the surface of sphere, Gaussians in high dimension, Random projection, Practical applications of t-SNE, Multidimensional Scaling, Manifold Learning, UMAP. Unit-IV: The General Models for Massive Data Problems: Topic Models - Non-Negative Matrix Factorization, Latent Dirichlet Allocation (LDA). Singular Value Decomposition, Dimensionality Reduction: Principal Component Analysis, Independent Component Analysis, Linear Discriminant Analysis and Practical implementation.",

    "Course: FUNDAMENTALS OF PROBABILITY AND STATISTICS | Code: 6.0MBD03 | Credits: 4 | Examination Scheme: Internal Assessment: 40 Marks (CIA-I: 20 Marks Unit I & II, CIA-II: 20 Marks Written Exams/ Quizzes/Assignment /Presentations/ Viva-Voce), End Semester Examination: 60 Marks (Unit I-IV). Unit-I: Data and its Types, Frequency table, histogram, measures of location, measures of spread, skewness, kurtosis, percentiles, box plot. Probability function, conditional probability, Bayes theorem, random variables, discrete and continuous random variables, discrete distributions: binomial, Poisson, geometric, hypergeometric, negative binomial, etc. Unit-II: Continuous distributions: uniform, normal, exponential, gamma, Weibull, Pareto, lognormal, Laplace, Cauchy, logistic distributions; properties and applications. Functions of random variables and their distributions using (i) transformation of rv (ii) MGF and (iii) method of Jacobian of transformation. Unit-III: Concept of a sampling distribution. Sampling distributions of t, chi-square and F (central and noncentral), their properties and applications. Goodness of fit and Categorical data analysis, Contingency Table, Measure of Associations. Test for Correlation. Unit-IV: Compound, truncated and mixture distributions. Convolutions of two distributions. Order statistics: their distributions and properties. Joint, marginal and conditional distribution of order statistics. The distribution of range and median. Convergence of RV, Central Limit Theorem, Weak law of large numbers.",

    "Course: LINEAR ALGEBRA AND ITS APPLICATIONS | Code: 6.0MBD04 | Credits: 3 | Examination Scheme: Internal Assessment: 40 Marks (CIA-I: 20 Marks Unit I & II, CIA-II: 20 Marks Written Exams/ Quizzes/Assignment /Presentations/Viva-Voce), End Semester Examination: 60 Marks (Unit I-IV). Unit-I: Matrices, System of Linear Equations, Elementary Row Operations, Row Reduced Matrices, Invertible Matrices. Fields, Vector spaces, Subspaces, Linear Combinations, Linear span, Linear dependence and Linear independence of vectors, Basis and Dimension, Ordered Basis, Finite dimensional vector spaces. Unit-II: Linear transformations and their representation as matrices, Kernel and Image of a linear transformation, Rank and Nullity Theorem, Change of Basis, Eigenvalues and Eigenvectors of a linear transformation (matrices), Eigendecomposition of symmetric matrices, Eigendecomposition of singular matrices, Characteristic polynomial and minimal polynomial, Diagonalization of linear operators, invariant subspaces, Jordan. Unit-III: Normed space, Inner product spaces, Cauchy-Schwarz-inequality, Orthogonal vectors, Orthonormal sets and bases, Projections, decompose vector to orthogonal components, Gram-Schmidt algorithm. Positive Definite and Positive Semi-definite Matrices, Rayleigh quotient, Linear functional on an Inner Product Space and Adjoint of a Linear operator, Self-adjoint operators, Normal Operators, Spectral theory of matrix, Spectral Decomposition, The Trace Operator. Quadratic forms. Unit-IV: Numerical methods: Bisection Method, Newton Raphson, Steepest Ascent, method of conjugate gradients; Direct and iterative methods for solving a linear system of equations: Gaussian elimination, LU factorization, Cholesky method, QR factorization, Householder's matrices, Jacobi's method, Gauss-Seidel method, successive over-relaxation methods (SOR), conditioning of a problem.",

    "Course: PROGRAMMING IN PYTHON | Code: 6.0MBD71 | Credits: 3 | Examination Scheme: Internal Assessment: 40 Marks (CIA-I: 20 Marks Unit I & II, CIA-II: 20 Marks Written Exams/ Quizzes/Assignment /Presentations/ Viva-Voce), End Semester Examination: 60 Marks (Unit I-IV). Unit-I: Introduction to Python: Basics of Python, Identifiers, literals, keywords, indentation, Strings, input and output statements, basic operators (arithmetic, logical, relational, bitwise and assignment), global variables, iterations-for, while, do-while, Control structures-exit function, break, continue, if-else. Unit-II: Functions and anonymous functions- built-in functions, user-defined functions, and lambda functions, recursion, scoping, and modules. Unit-III: Python Data structures: matrix, data frames, array, tuples, lists, dictionaries, lists, and mutability, string handling. Object-oriented programming, abstract data types, classes, inheritance, encapsulation. Unit-IV: File Handling and Exceptions Handling: Introduction to Files, modes for file opening, reading and writing structured and text files, closing and deletion of files, errors, and exceptions. Essential Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, SciPy, Statsmodels, Seaborn, etc.",

    "Course: PROFESSIONAL COMMUNICATION | Code: 6.0MBD81 | Credits: 2 | Examination Scheme: Internal Assessment: 40 Marks (CIA-I: 20 Marks Unit I & II, CIA-II: 20 Marks Written Exams/ Quizzes /Assignment/Presentations/ Viva-Voce), End Semester Examination: 60 Marks (Unit I-IV). Unit-I: Fundamentals of Communication: Definition, Process, Importance of Communication, Types of Communication, Purpose of Professional Communication, Barriers to Communication. Unit-II: Oral Communication: Listening Skill (Effective Listening, Intensive Listening vs Extensive listening, Techniques of Effective Listening, Listening and Note Taking), Speaking Skills (Paralinguistic features, Rate, Pauses, Volume, Pitch/Intonation/Voice Modulation, Pronunciation and Articulation), Activities (Group Discussions, Debates, Interviews, Public Speaking). Unit-III: Readings Skills (Effective Reading, Skimming, Scanning, Extensive Reading, Intensive Reading). Written communication: Academic Writing, Critical Thinking, Technical Writing vs Creative Writing, Paragraph Writing, Business Letters, Business Reports, Research Papers, Advertising, Notices, Emails, Resume writing, Cover Letter. Unit-IV: Soft Skills: Body Language (Personal Appearance, Gesture, posture, facial expression, eye contact), Proxemics/ Space Distance, Presentation Skills.",
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


# 3. Load Models Once (Auto-refreshes if documents change)
@st.cache_resource
def load_models(doc_list):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    doc_vectors = embed_model.encode(doc_list, normalize_embeddings=True)

    tokenized_docs = []
    for doc in doc_list:
        tokenized_docs.append(tokenize(doc))
    bm25 = BM25Okapi(tokenized_docs)

    return embed_model, rerank_model, doc_vectors, bm25


embed_model, rerank_model, doc_vectors, bm25 = load_models(documents)

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
