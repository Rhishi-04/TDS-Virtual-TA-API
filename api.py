import base64
import json
import pickle
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# === Load saved data and models at startup ===
EMBEDDING_DATA_FILE = "data/embedding_data.pkl"
FAISS_INDEX_FILE = "data/faiss_index.index"
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
GENERATION_MODEL_NAME = "google/flan-t5-large"

with open(EMBEDDING_DATA_FILE, "rb") as f:
    embedding_data = pickle.load(f)

index = faiss.read_index(FAISS_INDEX_FILE)

emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_model.to(device)

app = FastAPI()

# === Pydantic Models ===
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

# === Utility Functions ===
def normalize(v):
    return v / np.linalg.norm(v)

def retrieve(query, model, index, embedding_data, top_k=10):
    query_emb = normalize(model.encode(query, convert_to_numpy=True)).astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)
    results = []
    for i, idx in enumerate(I[0]):
        data = embedding_data[idx]
        data_with_score = data.copy()
        data_with_score["score"] = float(D[0][i])
        results.append(data_with_score)
    return results

from sentence_transformers.util import cos_sim

def rerank(query, posts, model):
    query_emb = model.encode(query, convert_to_tensor=True)
    for post in posts:
        post_emb = model.encode(post["combined_text"], convert_to_tensor=True)
        post["score"] = float(cos_sim(query_emb, post_emb))
    sorted_posts = sorted(posts, key=lambda x: x["score"], reverse=True)
    return sorted_posts


def truncate_context(texts, tokenizer, max_tokens=1500):
    total = 0
    selected = []
    for t in texts:
        tokens = tokenizer(t, truncation=True, return_tensors="pt")["input_ids"].shape[-1]
        if total + tokens > max_tokens:
            break
        selected.append(t)
        total += tokens
    return selected

def generate_answer(query, retrieved_texts, tokenizer, model, max_length=256):
    context = "\n\n".join(retrieved_texts)
    prompt = (
        "You are a knowledgeable assistant.\n"
        "Use the following context to answer the question in detail.\n"
        "Return ONLY a JSON object with two fields:\n"
        " - 'answer': a detailed, well-formed text answer.\n"
        " - 'links': a list of {'url': URL, 'text': Description} objects referring to relevant posts.\n"
        "Do NOT add any explanations outside the JSON.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer JSON:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=6,
        temperature=0.3,  # lower temp for more focused output
        early_stopping=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def extract_links_from_posts(posts):
    seen_urls = set()
    links = []
    for post in posts:
        url = post.get("url") or f"https://discourse.onlinedegree.iitm.ac.in/t/{post['topic_id']}"
        text_snippet = post.get("topic_title") or "Related discussion"
        if url not in seen_urls:
            links.append({"url": url, "text": text_snippet})
            seen_urls.add(url)
    return links


# === API Endpoint ===
@app.post("/api/", response_model=QueryResponse)
async def answer_question(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if req.image:
        try:
            image_bytes = base64.b64decode(req.image)
            # Optional OCR hook here
            # ocr_text = run_ocr(image_bytes)
            # question += "\n" + ocr_text
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    results = retrieve(question, emb_model, index, embedding_data)
    results = rerank(question, results, emb_model)

    retrieved_texts = [r["combined_text"] for r in results]
    truncated_texts = truncate_context(retrieved_texts, gen_tokenizer, max_tokens=1500)

    answer = generate_answer(question, truncated_texts, gen_tokenizer, gen_model)

    links = extract_links_from_posts(results)

    return QueryResponse(answer=answer.strip(), links=links)
