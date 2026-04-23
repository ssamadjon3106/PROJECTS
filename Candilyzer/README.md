# 🚀 Candilyzer — AI-Powered Technical Candidate Analyzer

Candilyzer is a structured, AI-driven system for evaluating software engineering candidates using real-world signals from GitHub and LinkedIn.

It performs **evidence-based technical assessment**, generates detailed reports, and ranks candidates with calibrated scoring — enabling faster and more reliable hiring decisions.

---

## ✨ Features

* 🧠 **Multi-stage AI pipeline**
  Fetch → Compress → Evaluate → Rank

* ⚡ **Fast & scalable**

  * Parallel data fetching
  * Parallel evaluation
  * Cached API responses

* 📊 **Calibrated scoring system**

  * Normalized scores (fair comparison)
  * Confidence scoring
  * Tie-breaking logic

* 🔥 **Streaming UI**

  * Candidate results appear instantly
  * No need to wait for full completion

* 🏆 **Decision-ready output**

  * Ranked candidate table
  * Automatic Top-K selection

---

## 🧠 Architecture

```
[GitHub API]   [Exa API]
      ↓             ↓
      └──→ Fetch Layer (parallel + cached)
                    ↓
          Compressor (fast LLM)
                    ↓
          Evaluator (Nemotron)
                    ↓
     Ranking + Calibration + Confidence
                    ↓
              Streamlit UI
```

---

## 📁 Project Structure

```
candilyzer/
│
├── app.py
└── src/
    ├── config.py        # global settings
    ├── fetcher.py       # data fetching + caching
    ├── compressor.py    # fast LLM summarization
    ├── evaluator.py     # main evaluation model
    ├── ranking.py       # scoring + calibration
    └── pipeline.py      # orchestration
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ssamadjon3106/candilyzer.git
cd candilyzer
```

---

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```
NEBIUS_API_KEY=your_nebius_key
GITHUB_API_KEY=your_github_token
EXA_API_KEY=your_exa_key
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🧪 How to Use

1. Enter a **full job description** (role, requirements, tech stack)
2. Input **GitHub usernames** (comma-separated)
3. Click **Analyze**

---

## 📊 Output

For each candidate:

* Technical evaluation report
* Strengths and weaknesses
* Score (0–100)

Final output includes:

* 📈 Ranked comparison table
* 🎯 Confidence scores
* ⭐ Top candidates selection

---

## 🧮 Scoring System

| Category                        | Weight |
| ------------------------------- | ------ |
| Technical Skills & Code Quality | 30     |
| Project Impact & Ownership      | 25     |
| Activity & Consistency          | 15     |
| Professional Credibility        | 20     |
| Role Fit                        | 10     |

Scores are:

* **Normalized** across candidates
* Adjusted using **confidence signals**
* Ranked using **tie-breaking logic**

---

## ⚡ Performance Optimizations

* Parallel execution (`ThreadPoolExecutor`)
* Cached GitHub & LinkedIn queries
* Token reduction via compression stage
* Separation of fast vs slow models

---

## 🧠 Models Used

* **Fast model (compression)** — lightweight, cost-efficient
* **Evaluation model** — reasoning-optimized for structured analysis

---

## ⚠️ Disclaimer

Candilyzer provides **AI-assisted evaluation**, not final hiring decisions.
Human review is required before making hiring choices.

---

## 🚀 Roadmap

* Interview question generation
* Role-specific scoring weights
* Redis-based caching
* Background job processing (Celery)
* API backend (FastAPI)
* Multi-role evaluation templates


---
### 🔗 Live demo
https://candilyzer-sss.streamlit.app


---
## 📄 License

MIT License
