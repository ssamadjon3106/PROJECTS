# Semantic Book Recommender 📚

An AI-powered book recommendation system that uses semantic search and LLM explanations to recommend books based on natural language queries.

The system combines:

* Sentence Transformers for semantic embeddings
* FAISS for fast vector similarity search
* Groq LLM for explanations
* Gradio for the interactive interface

Instead of keyword search, the system understands the *meaning* of a query.

Example query:

"Books about war strategy and leadership"

The system retrieves the most semantically similar books and explains why they match.

---

# Architecture

User Query
↓
Sentence Transformer Embedding
↓
FAISS Vector Search
↓
Top K Similar Books
↓
LLM Explanation (Groq)
↓
Gradio Interface

---

# Features

Semantic search across book descriptions
Vector database using FAISS
AI-generated explanations for recommendations
Interactive UI with Gradio
Fast retrieval using cached embeddings

---

# Project Structure

```
project/
│
├── app.py
├── books.csv
├── embeddings.npy
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```
git clone https://github.com/yourusername/book-recommender
cd book-recommender
```

Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Environment Variables

You must add your Groq API key.

```
export GROQ_API_KEY=your_api_key_here
```

Without this key the system will still work, but explanations will fall back to the book description.

---

# Run the Application

```
python app.py
```

Then open:

```
http://127.0.0.1:7860
```

---

# Dataset

The dataset must contain these columns:

```
title
authors
description
```

Optional columns:

```
thumbnail
categories
```

---

# Example Query

```
"Books about artificial intelligence and philosophy"
```

Result:

* semantic search retrieves relevant books
* LLM explains why they match the request

---

# Technologies Used

Sentence Transformers
FAISS Vector Search
Groq LLM API
Gradio UI

---

# Future Improvements

Add hybrid search (semantic + keyword)
Add filtering by genre and category
Store vectors in a real vector database
Add user personalization

---

# License

MIT License
