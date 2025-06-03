import os
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === Configuration ===
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-base"
TOP_K = 5
JSON_PATH = "data/discourse_posts.json"
TDS_MD_DIR = "tds_pages_md"
METADATA_PATH = "data/metadata.json"

# === Helper Functions ===
def clean_text(text):
    return " ".join(text.strip().split())

def normalize(v):
    return v / np.linalg.norm(v)

# === Load Data ===
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

# === Threading Logic ===
def build_reply_map(posts):
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        posts_by_number[post["post_number"]] = post
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

# === Embedding Preparation for discourse ===
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
            embedding_data.append({
                "topic_id": topic_id,
                "topic_title": title,
                "root_post_number": root_post["post_number"],
                "post_numbers": [p["post_number"] for p in subthread],
                "combined_text": combined_text,
                "source": "discourse",
            })
    return embedding_data, np.vstack(embeddings).astype("float32")

# === Load and parse markdown course content ===
def load_md_files(md_dir, metadata_path):
    md_pages = []
    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    # Create a mapping from filename to title
    filename_to_title = {}
    for entry in metadata:
        filename = entry.get("filename")
        title = entry.get("title")
        if filename and title:
            filename_to_title[filename] = title
    # Process markdown files
    for filename in os.listdir(md_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(md_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read()
            # Use title from metadata if available
            title = filename_to_title.get(filename, filename[:-3])  # Remove .md
            # Remove frontmatter block (between --- and ---)
            content_start = raw.find("---", 3)  # find second ---
            if content_start != -1:
                content = raw[content_start+3:].strip()
            else:
                content = raw
            md_pages.append({
                "title": title,
                "content": content
            })
    return md_pages

# === Embedding Preparation for course content ===
def prepare_course_content_embeddings(model, md_pages):
    embedding_data = []
    embeddings = []

    for page in tqdm(md_pages, desc="Embedding course content"):
        title = page.get("title", "Untitled")
        content = clean_text(page.get("content", ""))
        combined_text = f"Course Content Title: {title}\n\n{content}"
        emb = normalize(model.encode(combined_text, convert_to_numpy=True))
        embeddings.append(emb)
        embedding_data.append({
            "topic_id": None,
            "topic_title": title,
            "root_post_number": None,
            "post_numbers": [],
            "combined_text": combined_text,
            "source": "course_content",
        })

    return embedding_data, np.vstack(embeddings).astype("float32")

# === FAISS Setup ===
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
    prompt = f"Answer the question based on the following forum discussion excerpts and course content:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
    outputs = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Main Pipeline ===
if __name__ == "__main__":
    print("Loading discourse posts...")
    posts = load_posts(JSON_PATH)
    topics = group_posts_by_topic(posts)
    print(f"Loaded {len(posts)} posts across {len(topics)} topics.")

    print("Loading embedding model...")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Preparing discourse embeddings...")
    embedding_data_discourse, embeddings_discourse = prepare_embeddings(emb_model, topics)

    print("Loading markdown course content files...")
    md_pages = load_md_files(TDS_MD_DIR, METADATA_PATH)
    print(f"Loaded {len(md_pages)} markdown pages from course content.")

    print("Preparing course content embeddings...")
    embedding_data_course, embeddings_course = prepare_course_content_embeddings(emb_model, md_pages)

    # Combine discourse and course embeddings & data
    embedding_data_combined = embedding_data_discourse + embedding_data_course
    embeddings_combined = np.vstack([embeddings_discourse, embeddings_course])

    print("Building combined FAISS index...")
    index = build_faiss_index(embeddings_combined)
    print(f"Indexed {len(embedding_data_combined)} entries (discourse + course content).")

    # Example query
    query = "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?"

    print("Retrieving relevant content...")
    results = retrieve(query, emb_model, index, embedding_data_combined)

    for i, res in enumerate(results, 1):
        source = res.get("source", "unknown")
        print(f"\n[{i}] Score: {res['score']:.4f} | Source: {source} | Title: {res['topic_title']}")

    print("\nLoading generation model...")
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)

    retrieved_texts = [r["combined_text"] for r in results]
    answer = generate_answer(query, retrieved_texts, gen_tokenizer, gen_model)
    print("\nGenerated Answer:\n", answer)
