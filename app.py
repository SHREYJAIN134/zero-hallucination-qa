import streamlit as st
import torch
from pypdf import PdfReader
import docx
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

st.set_page_config(page_title="Zero-Hallucination Document QA", layout="wide")

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 1. Generator LLM
    gen_id = "Qwen/Qwen2.5-1.5B-Instruct"
    gen_tok = AutoTokenizer.from_pretrained(gen_id)
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_id, 
        torch_dtype=dtype, 
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    # 2. NLI Verifier
    nli_id = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_tok = AutoTokenizer.from_pretrained(nli_id)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_id).to(device)
    
    return gen_tok, gen_model, nli_tok, nli_model, device

gen_tok, gen_model, nli_tok, nli_model, device = load_models()

def check_entailment(context_chunk, hypothesis):
    inputs = nli_tok(context_chunk, hypothesis, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    return probs[0].item() >= 0.50

def zero_hallucination_engine(context, query):
    prompt_text = gen_tok.apply_chat_template([
        {
            "role": "system",
            "content": "You are an extractive, strictly grounded AI. Answer ONLY using direct facts from the context. If the text lacks the answer, reply EXACTLY with: 'info unavailable'."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nTask/Question:\n{query}"
        }
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = gen_tok(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=150, do_sample=False, temperature=0.0)
    
    raw_response = gen_tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    if "info unavailable" in raw_response.lower() or not check_entailment(context, raw_response):
        return "info unavailable"
        
    return raw_response

def extract_text(file_obj):
    name = file_obj.name.lower()
    text = ""
    if name.endswith(".pdf"):
        reader = PdfReader(file_obj)
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    elif name.endswith(".docx"):
        doc = docx.Document(file_obj)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    else:
        text = file_obj.read().decode("utf-8", errors="ignore")
    return text.strip()

# UI Layout
st.title("🛡️ Zero-Hallucination Document Summarizer & QA")
st.write("Strictly grounded extraction using **Qwen-2.5-1.5B** with **DeBERTa-v3 NLI Guardrail**.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    pasted_text = st.text_area("Or Paste Context", height=180)
    query = st.text_input("Enter Question or Summarization Task")
    submit = st.button("Analyze Document", type="primary")

with col2:
    st.subheader("Strictly Grounded Output")
    if submit:
        context = extract_text(uploaded_file) if uploaded_file else pasted_text
        if not context:
            st.warning("Please upload a file or enter context.")
        elif not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing and verifying facts..."):
                output = zero_hallucination_engine(context, query)
                st.info(output)