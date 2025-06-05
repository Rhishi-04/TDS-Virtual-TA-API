import os
import json
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Configuration ===
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
GENERATION_MODEL = "google/flan-t5-base"
TOP_K = 5
JSON_PATH = "data/discourse_posts.json"
MD_DIR = "tds_pages_md"
METADATA_PATH = "data/metadata.json"
BATCH_SIZE = 8
MIN_WORDS_PER_CONTENT = 20

# === Utility Functions ===
def clean_text(text):
    return " ".join(text.strip().split())

def normalize(v):
    return v / np.linalg.norm(v)

# === Load Course Markdown Content ===
def load_md_files(md_dir: str, metadata_path: str) -> List[Dict]:
    if not os.path.exists(md_dir):
        return []
    md_pages = []
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
            metadata = {entry.get("filename"): entry.get("title") for entry in metadata_list}
    for filename in os.listdir(md_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(md_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read()
            title = metadata.get(filename, filename[:-3])
            content_start = raw.find("---", 3)
            content = raw[content_start + 3:].strip() if content_start != -1 else raw
            md_pages.append({"title": title, "content": content})
    return md_pages

def prepare_course_content_embeddings(model: SentenceTransformer, md_pages: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
    texts, meta = [], []
    for page in md_pages:
        title = clean_text(page.get("title", "Untitled"))
        content = clean_text(page.get("content", ""))
        if not content or len(content.split()) < MIN_WORDS_PER_CONTENT:
            continue
        combined_text = f"Course Content Title: {title}\n\n{content}"
        texts.append(combined_text)
        meta.append({"topic_id": None, "topic_title": title, "root_post_number": None,
                     "post_numbers": [], "combined_text": combined_text,
                     "source": "course_content", "urls": []})
    if not texts:
        return [], np.array([])
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=BATCH_SIZE, show_progress_bar=True)
    embeddings = np.array([normalize(e) for e in embeddings], dtype=np.float32)
    return meta, embeddings

# === Forum Thread Processing ===
def load_posts(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def group_posts_by_topic(posts):
    topics = defaultdict(lambda: {"topic_title": "", "posts": []})
    for post in posts:
        topic_id = post["topic_id"]
        topics[topic_id]["topic_title"] = post.get("topic_title", "")
        topics[topic_id]["posts"].append(post)
    for topic in topics.values():
        topic["posts"].sort(key=lambda p: p["post_number"])
    return topics

def build_reply_map(posts):
    reply_map = defaultdict(list)
    posts_by_number = {post["post_number"]: post for post in posts}
    for post in posts:
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(root_post_number, reply_map, posts_by_number):
    collected = []
    def dfs(post_num):
        post = posts_by_number[post_num]
        collected.append(post)
        for child in reply_map.get(post_num, []):
            dfs(child["post_number"])
    dfs(root_post_number)
    return collected

def prepare_embeddings(model, topics):
    embedding_data, embeddings = [], []
    for topic_id, data in tqdm(topics.items(), desc="Embedding threads"):
        posts = data["posts"]
        title = data["topic_title"]
        reply_map, posts_by_number = build_reply_map(posts)
        for root_post in reply_map[None]:
            subthread = extract_subthread(root_post["post_number"], reply_map, posts_by_number)
            combined_text = f"Topic title: {title}\n\n" + "\n\n---\n\n".join(clean_text(p["content"]) for p in subthread)
            emb = normalize(model.encode(combined_text, convert_to_numpy=True))
            embeddings.append(emb)
            embedding_data.append({"topic_id": topic_id, "topic_title": title,
                                   "root_post_number": root_post["post_number"],
                                   "post_numbers": [p["post_number"] for p in subthread],
                                   "combined_text": combined_text})
    return embedding_data, np.vstack(embeddings).astype("float32")

# === Retrieval & FAISS ===
def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, model, index, embedding_data, top_k=TOP_K):
    query_emb = normalize(model.encode(query, convert_to_numpy=True)).astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)
    return [embedding_data[idx] | {"score": float(D[0][i])} for i, idx in enumerate(I[0])]

# === Answer Generation ===
def generate_answer(query, retrieved_texts, tokenizer, model, max_length=256):
    context = "\n\n".join(retrieved_texts)
    prompt = f"Answer the question based on the following excerpts:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
    outputs = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Main Pipeline ===
if __name__ == "__main__":
    print("Loading forum posts...")
    posts = load_posts(JSON_PATH)
    topics = group_posts_by_topic(posts)

    print("Loading embedding model...")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Preparing forum thread embeddings...")
    embedding_data_f, embeddings_f = prepare_embeddings(emb_model, topics)

    print("Preparing course content embeddings...")
    md_pages = load_md_files(MD_DIR, METADATA_PATH)
    embedding_data_c, embeddings_c = prepare_course_content_embeddings(emb_model, md_pages)

    embedding_data = embedding_data_f + embedding_data_c
    embeddings = np.vstack([embeddings_f, embeddings_c])

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    import pickle
    os.makedirs("data", exist_ok=True)

    with open("data/embedding_data.pkl", "wb") as f:
        pickle.dump(embedding_data, f)
    print("✅ Saved: data/embedding_data.pkl")

    faiss.write_index(index, "data/faiss_index.index")
    print("✅ Saved: data/faiss_index.index")

    query = "How to get Dummy API key for GA3"

    print("Retrieving answers...")
    results = retrieve(query, emb_model, index, embedding_data)
    for i, res in enumerate(results, 1):
        print(f"[{i}] Score: {res['score']:.4f} | Source: {res.get('source', 'forum')} | Title: {res['topic_title']}")

    print("\nLoading generation model...")
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)

    retrieved_texts = [r["combined_text"] for r in results]
    answer = generate_answer(query, retrieved_texts, gen_tokenizer, gen_model)
    print("\nGenerated Answer:\n", answer)
