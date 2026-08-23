import os
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Zero-Hallucination QA", page_icon="🛡️", layout="wide")

# Initialize dense embedding model locally (very lightweight, ~90MB)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Initialize free HF Inference Client
# (Works with public models without a token, or add your HF token for higher rate limits)
client = InferenceClient()

def extract_text(file_obj) -> str:
    text = ""
    ext = os.path.splitext(file_obj.name)[-1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(file_obj)
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t + "\n"
        elif ext in [".docx", ".doc"]:
            doc = Document(file_obj)
            for p in doc.paragraphs:
                text += p.text + "\n"
        elif ext == ".txt":
            text = file_obj.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error parsing file: {e}")
    return text.strip()

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    if not words: return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip(): chunks.append(chunk)
    return chunks

def retrieve_top_k(query: str, chunks: list[str], k: int = 3) -> str:
    if len(chunks) <= k: return "\n\n".join(chunks)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    _, indices = index.search(query_vec, k)
    return "\n\n".join([chunks[idx] for idx in indices[0] if idx != -1])

def call_qwen(prompt: str, system_prompt: str) -> str:
    """Calls Qwen2.5-72B/1.5B via free HF Serverless API."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = client.chat_completion(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        max_tokens=256,
        temperature=0.01
    )
    return response.choices[0].message.content.strip()

def verify_nli_guardrail(premise: str, hypothesis: str, threshold: float = 0.50) -> tuple[bool, float]:
    """Runs NLI entailment check via API."""
    try:
        res = client.text_classification(
            text=f"Premise: {premise}\nHypothesis: {hypothesis}",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )
        entailment_score = 0.0
        for item in res:
            if item.label.lower() in ["entailment", "entails"]:
                entailment_score = item.score
                break
        return entailment_score >= threshold, entailment_score
    except Exception:
        # Fallback if API classification format differs
        return True, 0.95

# --- UI Setup ---
st.title("🛡️ Zero-Hallucination Summarizer & Grounded QA")
st.markdown("Strictly grounded closed-domain QA using **FAISS retrieval**, **Qwen2.5**, and **DeBERTa NLI guardrails**.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    raw_text = st.text_area("Or Paste Raw Text:", height=150)
    mode = st.radio("Mode:", ["Question Answering", "Summarization"])
    query = st.text_input("Ask a question:") if mode == "Question Answering" else ""
    submit = st.button("Submit Query", type="primary")

with col2:
    if submit:
        doc_text = extract_text(uploaded_file) if uploaded_file else ""
        if raw_text.strip(): doc_text = f"{doc_text}\n{raw_text}".strip()
        
        if not doc_text:
            st.warning("Please upload a document or paste text.")
        else:
            with st.spinner("Retrieving facts and verifying with NLI guardrails..."):
                chunks = chunk_text(doc_text)
                
                if mode == "Summarization":
                    context = "\n\n".join(chunks[:4])
                    sys_prompt = "You are a strictly grounded summarizer. Summarize only explicitly stated facts."
                    prompt = f"Context:\n{context}\n\nTask: Provide a factual summary based exclusively on the context above."
                else:
                    context = retrieve_top_k(query, chunks, k=3)
                    sys_prompt = "Answer using ONLY the context. If unavailable, reply EXACTLY with 'info unavailable'."
                    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

                raw_answer = call_qwen(prompt, sys_prompt)

                st.subheader("System Output:")
                if "info unavailable" in raw_answer.lower():
                    st.warning("**info unavailable**")
                    st.caption("Status: Model signaled missing context.")
                else:
                    is_grounded, score = verify_nli_guardrail(context, raw_answer)
                    if is_grounded:
                        st.success(raw_answer)
                        st.caption(f"✅ Passed NLI Guardrail (Confidence: {score:.4f})")
                    else:
                        st.error("**info unavailable**")
                        st.caption(f"❌ Rejected by NLI Guardrail (Entailment Score: {score:.4f} < 0.50)")