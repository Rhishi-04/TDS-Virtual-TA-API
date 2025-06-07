import base64
import io
import json
import pickle
import logging
import os
from datetime import datetime
from typing import Optional, List

import faiss
import numpy as np
from PIL import Image
import pytesseract
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI, OpenAIError

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Config ===
EMBEDDING_DATA_FILE = "data/embedding_data.pkl"
FAISS_INDEX_FILE = "data/faiss_index.index"
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Must be set in environment

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

# === Load Data and Models ===
with open(EMBEDDING_DATA_FILE, "rb") as f:
    embedding_data = pickle.load(f)
index = faiss.read_index(FAISS_INDEX_FILE)

emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# === Pydantic Schemas ===
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 image

class Link(BaseModel):
    url: Optional[str]
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]
    confidence: float
    timestamp: str

# === Utility Functions ===
def normalize(v):
    return v / np.linalg.norm(v)

def rerank(query, docs, model):
    """Re-rank retrieved docs using cross-encoder similarity"""
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = [model.encode(doc["combined_text"], convert_to_tensor=True) for doc in docs]
    scores = [util.pytorch_cos_sim(query_emb, d_emb).item() for d_emb in doc_embs]
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = score
    docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return docs

def truncate_context(texts, max_tokens=1500):
    """Truncate concatenated context to fit max token length"""
    total = 0
    truncated = []
    for text in texts:
        tokens = len(text.split())
        if total + tokens > max_tokens:
            break
        truncated.append(text)
        total += tokens
    return truncated

def retrieve(query, model, index, embedding_data, top_k=10):
    query_emb = normalize(model.encode(query, convert_to_numpy=True)).astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)
    results = []
    for i, idx in enumerate(I[0]):
        data = embedding_data[idx]
        data_with_score = data.copy()
        data_with_score["score"] = float(D[0][i])
        results.append(data_with_score)
    # Re-rank top_k results
    reranked = rerank(query, results, model)
    return reranked, reranked[0]["rerank_score"] if reranked else 0.0

def extract_text_from_base64(base64_str):
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""

def build_prompt(query, retrieved_texts):
    context = "\n\n".join(retrieved_texts)
    prompt = (
        "You are a helpful assistant.\n"
        "Provide your answer ONLY in this exact JSON format:\n"
        "{\n"
        "  \"answer\": \"detailed answer\",\n"
        "  \"links\": [\n"
        "    { \"url\": \"https://example.com\", \n"
        "     \"text\": \"link description\" }\n"
        "  ]\n"
        "}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer JSON:"
    )
    return prompt

def query_openai(prompt):
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        logger.error(f"OpenAI API failed: {e}")
        return ""

def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("OpenAI response was not valid JSON.")
        return {"answer": "Error: Model failed to generate a proper answer.", "links": []}

# === Main API Endpoint ===
@app.post("/api/", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    query = req.question.strip()
    if req.image:
        query += " " + extract_text_from_base64(req.image)

    retrieved_docs, top_score = retrieve(query, emb_model, index, embedding_data, top_k=10)
    retrieved_texts = truncate_context([d["combined_text"] for d in retrieved_docs])
    links = [Link(url=d.get("url"), text=d.get("title", "More info")) for d in retrieved_docs]

    prompt = build_prompt(query, retrieved_texts)
    raw_response = query_openai(prompt)
    parsed = parse_json_response(raw_response)

    return QueryResponse(
        answer=parsed.get("answer", "Could not generate a reliable answer."),
        links=parsed.get("links", []) or links[:2],
        confidence=round(top_score, 4),
        timestamp=datetime.utcnow().isoformat()
    )
