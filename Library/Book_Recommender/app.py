import os
import numpy as np
import pandas as pd
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from groq import Groq

# =========================
# Configuration
# =========================

DATA_PATH = "books.csv"
EMBED_FILE = "embeddings.npy"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 12

# Secure API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# =========================
# Load Dataset
# =========================

df = pd.read_csv(DATA_PATH)

required_cols = ["title", "authors", "description"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"books.csv must contain column: {col}")

if "thumbnail" not in df.columns:
    df["thumbnail"] = ""

if "categories" not in df.columns:
    df["categories"] = ""

df = df.dropna(subset=["description"]).reset_index(drop=True)

# =========================
# Embedding Model
# =========================

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

descriptions = df["description"].tolist()

# =========================
# Embeddings
# =========================

if os.path.exists(EMBED_FILE):

    print("Loading cached embeddings...")
    embeddings_np = np.load(EMBED_FILE)

else:

    print("Encoding descriptions...")
    embeddings = embedder.encode(descriptions, show_progress_bar=True)

    embeddings_np = np.array(embeddings).astype("float32")

    embeddings_np = embeddings_np / np.linalg.norm(
        embeddings_np, axis=1, keepdims=True
    )

    np.save(EMBED_FILE, embeddings_np)

# =========================
# FAISS Index
# =========================

print("Building FAISS index...")

dim = embeddings_np.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings_np)

print(f"Indexed {index.ntotal} books")

# =========================
# Retrieval
# =========================

def retrieve_books(query, k=TOP_K):

    q_vec = embedder.encode([query]).astype("float32")

    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)

    distances, indices = index.search(q_vec, k)

    rows = df.iloc[indices[0]].copy()

    return rows


# =========================
# AI Explanation
# =========================

def explain_book(query, book):

    if not client:
        return book["description"]

    prompt = f"""
You are an AI librarian.

Explain why the following book matches the user request.

User request:
{query}

Book:
Title: {book['title']}
Author: {book['authors']}
Description: {book['description']}

Explain briefly.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return response.choices[0].message.content


# =========================
# Recommendation Pipeline
# =========================

current_rows = None
current_query = ""


def recommend(query, genre, category):

    global current_rows
    global current_query

    current_query = query

    rows = retrieve_books(query)

    current_rows = rows

    gallery = []

    for _, r in rows.iterrows():

        img = r["thumbnail"]

        if not isinstance(img, str) or img.strip() == "":
            img = "https://via.placeholder.com/150"

        gallery.append((img, r["title"]))

    return gallery


# =========================
# Click Book
# =========================

def show_book(evt: gr.SelectData):

    index = evt.index

    book = current_rows.iloc[index]

    img = book["thumbnail"]

    if not isinstance(img, str) or img.strip() == "":
        img = "https://via.placeholder.com/300"

    explanation = explain_book(current_query, book)

    description = f"""
## {book['title']}

**Author:** {book['authors']}

{explanation}
"""

    return img, description


# =========================
# Gradio UI
# =========================

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:

    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():

        description = gr.Textbox(
            label="Describe a book",
            placeholder="e.g. war strategy, love story, philosophy",
            scale=4
        )

        genre = gr.Dropdown(
            [
                "All",
                "History",
                "Fantasy",
                "Science Fiction",
                "Mystery",
                "Romance",
                "Philosophy",
                "Self Development",
                "Biography",
                "Politics",
                "Technology"
            ],
            value="All",
            label="Genre"
        )

        category = gr.Dropdown(
            [
                "All",
                "World War I",
                "World War II",
                "Ancient History",
                "Space Exploration",
                "Artificial Intelligence",
                "Startup",
                "Psychology",
                "Economics",
                "Crime Investigation"
            ],
            value="All",
            label="Category"
        )

        button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")

    gallery = gr.Gallery(
        columns=8,
        height=350,
        object_fit="cover"
    )

    selected_book = gr.Image(height=420)

    book_description = gr.Markdown()

    button.click(
        fn=recommend,
        inputs=[description, genre, category],
        outputs=gallery
    )

    gallery.select(
        fn=show_book,
        outputs=[selected_book, book_description]
    )

# =========================
# Launch
# =========================

demo.launch()
