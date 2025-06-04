import os
import json
import re
import logging
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.decomposition import PCA

# === Config ===
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
GENERATION_MODEL = "google/flan-t5-base"  # Smaller model
TOP_K = 5
BATCH_SIZE = 16  # Reduced for lower memory usage
REDUCED_DIM = 128  # Reduce embedding dimensionality

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "data/discourse_posts.json")
TDS_MD_DIR = os.path.join(BASE_DIR, "tds_pages_md")
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata.json")
EMBEDDING_DATA_PATH = os.path.join(BASE_DIR, "data/embedding_data.pkl")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data/faiss_index_reduced.index")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())
    return " ".join(text.strip().split())

def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm < eps else v / norm

def reduce_dimensionality(embeddings: np.ndarray, n_components: int = REDUCED_DIM) -> np.ndarray:
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings.astype(np.float32)

def load_json_file(path: str) -> List[Dict]:
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {path}: {e}")
        return []

def group_posts_by_topic(posts: List[Dict]) -> Dict[int, Dict]:
    topics = defaultdict(lambda: {"topic_title": "", "posts": []})
    for post in posts:
        topic_id = post.get("topic_id")
        if topic_id is None:
            logger.warning("Post missing topic_id, skipping.")
            continue
        topics[topic_id]["topic_title"] = post.get("topic_title", "")
        topics[topic_id]["posts"].append(post)
    for topic in topics.values():
        topic["posts"].sort(key=lambda p: p.get("post_number", 0))
    return topics

def build_reply_map(posts: List[Dict]) -> Tuple[Dict[Optional[int], List[Dict]], Dict[int, Dict]]:
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        post_num = post.get("post_number")
        if post_num is None:
            logger.warning(f"Post missing post_number, skipping: {post.get('content', '')[:50]}...")
            continue
        posts_by_number[post_num] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(
    root_post_number: int,
    reply_map: Dict[Optional[int], List[Dict]],
    posts_by_number: Dict[int, Dict]
) -> List[Dict]:
    collected = []
    def dfs(post_num):
        post = posts_by_number.get(post_num)
        if not post:
            return
        collected.append(post)
        for child in reply_map.get(post_num, []):
            dfs(child.get("post_number"))
    dfs(root_post_number)
    return collected

def prepare_embeddings(
    model: SentenceTransformer,
    topics: Dict[int, Dict],
    batch_size: int = BATCH_SIZE,
) -> Tuple[List[Dict], np.ndarray]:
    texts = []
    meta = []
    for topic_id, data in tqdm(topics.items(), desc="Preparing discourse embeddings"):
        posts = data.get("posts", [])
        title = clean_text(data.get("topic_title", ""))
        if not posts or not title:
            continue
        reply_map, posts_by_number = build_reply_map(posts)
        for root_post in reply_map.get(None, []):
            root_post_num = root_post.get("post_number")
            if not root_post_num:
                continue
            subthread = extract_subthread(root_post_num, reply_map, posts_by_number)
            post_texts = [clean_text(p.get("content", "")) for p in subthread]
            post_texts = [t for t in post_texts if t and len(t.split()) > 5]
            if not post_texts:
                continue
            combined_text = f"Topic title: {title}\n\n" + "\n\n---\n\n".join(post_texts)
            texts.append(combined_text)
            meta.append({
                "topic_id": topic_id,
                "topic_title": title,
                "root_post_number": root_post_num,
                "post_numbers": [p.get("post_number") for p in subthread],
                "combined_text": combined_text,
                "source": "discourse",
                "urls": [p.get("url") for p in subthread if p.get("url")]
            })
    if not texts:
        logger.warning("No valid discourse texts to embed.")
        return [], np.array([])
    logger.info(f"Encoding {len(texts)} discourse texts with batch size {batch_size}...")
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.array([normalize(e) for e in embeddings], dtype=np.float32)
    return meta, embeddings

def load_md_files(md_dir: str, metadata_path: str) -> List[Dict]:
    if not os.path.exists(md_dir):
        logger.error(f"Markdown directory not found: {md_dir}")
        return []
    md_pages = []
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)
                metadata = {entry.get("filename"): entry.get("title") for entry in metadata_list}
        except Exception as e:
            logger.warning(f"Could not load metadata JSON: {e}")
    for filename in os.listdir(md_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(md_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = f.read()
                title = metadata.get(filename, filename[:-3])
                content_start = raw.find("---", 3)
                content = raw[content_start + 3:].strip() if content_start != -1 else raw
                md_pages.append({
                    "title": title,
                    "content": content,
                })
            except Exception as e:
                logger.error(f"Error reading markdown file {filename}: {e}")
    logger.info(f"Loaded {len(md_pages)} markdown pages.")
    return md_pages

def prepare_course_content_embeddings(
    model: SentenceTransformer,
    md_pages: List[Dict],
    batch_size: int = BATCH_SIZE,
) -> Tuple[List[Dict], np.ndarray]:
    texts = []
    meta = []
    for page in md_pages:
        title = clean_text(page.get("title", "Untitled"))
        content = clean_text(page.get("content", ""))
        if not content or len(content.split()) < 10:
            continue
        combined_text = f"Course Content Title: {title}\n\n{content}"
        texts.append(combined_text)
        meta.append({
            "topic_id": None,
            "topic_title": title,
            "root_post_number": None,
            "post_numbers": [],
            "combined_text": combined_text,
            "source": "course_content",
            "urls": []
        })
    if not texts:
        logger.warning("No valid course content texts to embed.")
        return [], np.array([])
    logger.info(f"Encoding {len(texts)} course content pages...")
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.array([normalize(e) for e in embeddings], dtype=np.float32)
    return meta, embeddings

def build_faiss_index(embeddings: np.ndarray, nlist: int = 50) -> faiss.Index:
    if embeddings.size == 0:
        logger.error("Empty embeddings array, cannot build FAISS index.")
        raise ValueError("Empty embeddings array")
    embeddings = reduce_dimensionality(embeddings, n_components=REDUCED_DIM)
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 5
    logger.info(f"FAISS index built with dimension {dim} and {embeddings.shape[0]} vectors.")
    return index

def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    embedding_data: List[Dict],
    top_k: int = TOP_K,
    threshold: float = 0.6
) -> List[Dict]:
    query_emb = normalize(model.encode(query, convert_to_numpy=True)).astype(np.float32)
    D, I = index.search(np.array([query_emb]), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if score < threshold:
            continue
        data = embedding_data[idx].copy()
        data["score"] = float(score)
        results.append(data)
    return results

def clean_excerpt(text: str) -> str:
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[-=~]{3,}", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def generate_answer(
    query: str,
    retrieved: List[Dict],
    tokenizer,
    model,
    max_length: int = 256,
) -> Dict:
    retrieved_texts = [r["combined_text"] for r in retrieved]
    cleaned_texts = [clean_excerpt(t) for t in retrieved_texts[:5]]
    context = "\n\n".join(cleaned_texts)
    prompt = (
        "You are an expert teaching assistant for the Tools in Data Science course. "
        "Provide a clear, concise, and accurate answer to the question below using the given context. "
        "If the context is insufficient, state that and suggest checking the course materials or contacting the teaching team.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        length_penalty=1.0,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    all_links = []
    for res in retrieved[:2]:
        if res["source"] == "discourse":
            for url in res.get("urls", [])[:2]:
                post_num = res["post_numbers"][res["urls"].index(url)]
                post = next(p for p in topics[res["topic_id"]]["posts"] if p.get("post_number") == post_num)
                all_links.append({
                    "url": url,
                    "text": clean_text(post.get("content", ""))[:100] + "..." if len(post.get("content", "")) > 100 else clean_text(post.get("content", ""))
                })
    return {"answer": answer, "links": all_links[:2]}

def main():
    global topics
    logger.info("Loading discourse posts...")
    posts = load_json_file(JSON_PATH)
    topics = group_posts_by_topic(posts)
    if not topics:
        logger.error("No topics loaded, exiting.")
        return
    logger.info(f"Loaded {len(posts)} posts across {len(topics)} topics.")

    logger.info("Loading embedding model...")
    emb_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # Check if embeddings and FAISS index exist
    if os.path.exists(EMBEDDING_DATA_PATH) and os.path.exists(FAISS_INDEX_PATH):
        logger.info("Loading precomputed embeddings and FAISS index...")
        with open(EMBEDDING_DATA_PATH, "rb") as f:
            embedding_data_combined = pickle.load(f)
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        logger.info("Preparing discourse embeddings...")
        embedding_data_discourse, embeddings_discourse = prepare_embeddings(emb_model, topics)

        logger.info("Loading markdown course content files...")
        md_pages = load_md_files(TDS_MD_DIR, METADATA_PATH)

        logger.info("Preparing course content embeddings...")
        embedding_data_course, embeddings_course = prepare_course_content_embeddings(emb_model, md_pages)

        embedding_data_combined = embedding_data_discourse + embedding_data_course
        embeddings_combined = np.vstack([embeddings_discourse, embeddings_course]) if embeddings_discourse.size and embeddings_course.size else np.array([])

        if embeddings_combined.size == 0:
            logger.error("No embeddings generated, exiting.")
            return

        logger.info("Building combined FAISS index...")
        index = build_faiss_index(embeddings_combined)

        # Save embeddings and index to disk
        with open(EMBEDDING_DATA_PATH, "wb") as f:
            pickle.dump(embedding_data_combined, f)
        faiss.write_index(index, FAISS_INDEX_PATH)

    logger.info("Loading generation model...")
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL).to(device)
    gen_model.eval()
    gen_model = torch.quantization.quantize_dynamic(
        gen_model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Test query
    query = "How to get the dummy API key for GA3?"
    logger.info(f"Retrieving relevant content for query: {query}")
    results = retrieve(query, emb_model, index, embedding_data_combined)

    for i, res in enumerate(results, 1):
        snippet = res["combined_text"][:200].replace("\n", " ") + "..."
        logger.info(f"[{i}] Score: {res['score']:.4f} | Source: {res['source']} | Title: {res['topic_title']}")
        logger.info(f"Snippet: {snippet}")

    logger.info("Generating answer...")
    result = generate_answer(query, results, gen_tokenizer, gen_model)
    logger.info(f"Generated answer:\n{json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()
