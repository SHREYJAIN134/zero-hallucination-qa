import os
import torch
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Zero-Hallucination QA", page_icon="🛡️", layout="wide")

torch.set_num_threads(2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Cache models to load only once
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    
    gen_id = "Qwen/Qwen2.5-1.5B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_id)
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_id, torch_dtype=DTYPE, device_map=DEVICE, low_cpu_mem_usage=True
    )
    
    nli_id = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_id)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        nli_id, torch_dtype=DTYPE, device_map=DEVICE
    )
    return embedder, gen_tokenizer, gen_model, nli_tokenizer, nli_model

embedder, gen_tokenizer, gen_model, nli_tokenizer, nli_model = load_models()

def extract_text(file_obj) -> str:
    text = ""
    ext = os.path.splitext(file_obj.name)[-1].lower()
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

def verify_entailment(premise: str, hypothesis: str, threshold: float = 0.50) -> tuple[bool, float]:
    inputs = nli_tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = nli_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    prob = float(probs[0].item())
    return prob >= threshold, prob

# UI Layout
st.title("🛡️ Zero-Hallucination Summarizer & Grounded QA")
st.markdown("Strictly grounded closed-domain QA using **FAISS**, **Qwen2.5-1.5B**, and **DeBERTa NLI**.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    raw_text = st.text_area("Or paste text here:", height=150)
    mode = st.radio("Select Mode:", ["Question Answering", "Summarization"])
    query = st.text_input("Enter your question:") if mode == "Question Answering" else ""
    run_button = st.button("Submit Query", type="primary")

with col2:
    if run_button:
        doc_text = extract_text(uploaded_file) if uploaded_file else ""
        if raw_text.strip(): doc_text = f"{doc_text}\n{raw_text}".strip()
        
        if not doc_text:
            st.warning("Please upload a file or paste text.")
        else:
            with st.spinner("Processing through NLI Guardrail..."):
                chunks = chunk_text(doc_text)
                if mode == "Summarization":
                    retrieved_context = "\n\n".join(chunks[:4])
                    sys_prompt = "You are a strictly grounded summarizer. Summarize only explicitly stated facts."
                    prompt = f"Context:\n{retrieved_context}\n\nTask: Provide a factual summary based exclusively on the context above."
                else:
                    retrieved_context = retrieve_top_k(query, chunks, k=3)
                    sys_prompt = "Answer the question using ONLY the provided context. If unavailable, reply EXACTLY with 'info unavailable'."
                    prompt = f"Context:\n{retrieved_context}\n\nQuestion: {query}\nAnswer:"

                messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
                formatted_prompt = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = gen_tokenizer([formatted_prompt], return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = gen_model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=gen_tokenizer.eos_token_id)
                
                gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, outputs)]
                raw_ans = gen_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

                if "info unavailable" in raw_ans.lower():
                    st.subheader("System Output:")
                    st.write("**info unavailable**")
                    st.info("Status: Model signaled missing context.")
                else:
                    is_grounded, prob = verify_entailment(retrieved_context, raw_ans)
                    st.subheader("System Output:")
                    if is_grounded:
                        st.success(raw_ans)
                        st.caption(f"✅ Passed NLI Guardrail (Entailment Score: {prob:.4f})")
                    else:
                        st.error("**info unavailable**")
                        st.caption(f"❌ Rejected by NLI Guardrail (Score: {prob:.4f} < 0.50)")