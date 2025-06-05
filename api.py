import base64
import io
import json
import pickle
import logging
from datetime import datetime
from typing import Optional, List
import faiss
import numpy as np
import torch
from PIL import Image
import pytesseract
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
EMBEDDING_DATA_FILE = "data/embedding_data.pkl"
FAISS_INDEX_FILE = "data/faiss_index.index"
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
GENERATION_MODEL_NAME = "google/flan-t5-base"

# === Load Models and Data ===
with open(EMBEDDING_DATA_FILE, "rb") as f:
    embedding_data = pickle.load(f)
index = faiss.read_index(FAISS_INDEX_FILE)

emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME).to(device)
gen_model.eval()

app = FastAPI()

# === Pydantic Schemas ===
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

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

def rerank(query, posts, model):
    query_emb = model.encode(query, convert_to_tensor=True)
    for post in posts:
        post_emb = model.encode(post["combined_text"], convert_to_tensor=True)
        post["score"] = float(util.cos_sim(query_emb, post_emb))
    return sorted(posts, key=lambda x: x["score"], reverse=True)

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
        "You MUST answer ONLY in strict JSON format with the following schema:\n"
        "{\n"
        "  \"answer\": \"a detailed, well-formed text answer\",\n"
        "  \"links\": [\n"
        "    { \"url\": \"https://example.com\",\n"
        "      \"text\": \"Description of the link\" }\n"
        "  ]\n"
        "}\n\n"
        "Do not return markdown, natural language, or any extra commentary.\n"
        "Your ENTIRE OUTPUT must be a valid JSON object matching the schema.\n"
        "Do not add 'Answer:', 'Output:', or anything else before or after the JSON.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer JSON:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=6,
        temperature=0.3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_ocr(image_bytes: bytes) -> str:
    try:
        logger.info("Running OCR on image...")
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        logger.info("OCR completed successfully.")
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

def extract_links_from_posts(posts):
    seen_urls = set()
    links = []
    for post in posts:
        url = post.get("url") or f"https://discourse.onlinedegree.iitm.ac.in/t/{post['topic_id']}"
        text = post.get("topic_title", "Related discussion")
        if url not in seen_urls:
            links.append({"url": url, "text": text})
            seen_urls.add(url)
    return links

def calculate_confidence(results):
    if not results:
        return 0.0
    top_score = results[0].get("score", 0)
    return round(min(top_score, 1.0), 3)

# === API Endpoint ===
@app.post("/api/", response_model=QueryResponse)
async def answer_question(req: QueryRequest):
    start_time = datetime.utcnow()
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    image_text = ""
    if req.image:
        try:
            image_bytes = base64.b64decode(req.image)
            image_text = run_ocr(image_bytes)
            logger.info(f"Extracted text from image: {image_text[:50]}...")
        except base64.binascii.Error:
            logger.warning("Invalid base64 image data received.")
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    full_query = f"{question}\n\nImage context: {image_text}" if image_text else question

    results = retrieve(full_query, emb_model, index, embedding_data)
    if not results:
        return QueryResponse(
            answer="No relevant information found.",
            links=[],
            confidence=0.0,
            timestamp=start_time.isoformat()
        )

    results = rerank(full_query, results, emb_model)
    retrieved_texts = [r["combined_text"] for r in results]
    truncated_texts = truncate_context(retrieved_texts, gen_tokenizer)

    answer_json_str = generate_answer(full_query, truncated_texts, gen_tokenizer, gen_model)

    logger.error(f"Raw model output: {answer_json_str}")
    
    import re

    # Attempt to extract JSON block if there's extra text
    json_pattern = re.compile(r'\{.*\}', re.DOTALL)
    json_match = json_pattern.search(answer_json_str)
    if json_match:
        json_str_clean = json_match.group(0)
    else:
        logger.error(f"No JSON object found in output: {answer_json_str}")
        json_str_clean = "{}"

    logger.info(f"Cleaned JSON: {json_str_clean}")


    try:
        answer_data = json.loads(json_str_clean)
        answer = answer_data.get("answer", "Error: Model failed to generate a proper answer.")
        model_links = answer_data.get("links", [])
    except json.JSONDecodeError:
        logger.error(f"Failed to parse model output as JSON: {answer_json_str}")
        answer = "Error: Model output was not in the expected JSON format."
        model_links = []

    retrieved_links = extract_links_from_posts(results)

    final_links = []
    seen_urls = set()
    for link in model_links + retrieved_links:
        url = link.get("url")
        if url and url not in seen_urls:
            final_links.append(Link(url=url, text=link.get("text", "Related link")))
            seen_urls.add(url)

    confidence = calculate_confidence(results)
    return QueryResponse(
        answer=answer.strip(),
        links=final_links[:3],
        confidence=confidence,
        timestamp=start_time.isoformat()
    )

handler = app