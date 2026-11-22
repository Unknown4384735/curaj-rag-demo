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
    total =
