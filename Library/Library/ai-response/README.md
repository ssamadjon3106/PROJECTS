# AI Legal Agent Uzbekistan

AI Legal Agent Uzbekistan is a Streamlit app that analyzes legal-style requests about bank secrecy, personal data, and tax-related disclosures in an Uzbek legal context.

It combines:

- a synthetic local legal knowledge base,
- TF-IDF + FAISS retrieval,
- risk and compliance scoring,
- optional OpenAI generation,
- SQLite-backed interaction history and review workflow.

> Important: the bundled legal corpus is synthetic and intended for prototyping, demos, and workflow testing. It is not a substitute for real legal advice or official legal sources.

## What the app does

- Classifies a request by authority, intent, entities, and sensitive issues.
- Retrieves relevant legal context from `legal_docs/`.
- Generates a formal Uzbek response using OpenAI when configured, or a local fallback when not.
- Scores the output for risk, compliance, and confidence.
- Routes the result to auto-reply, bank review, or mandatory human review.
- Stores each interaction in SQLite for later inspection and approval/rejection.

## Project Structure

- `app.py` - Streamlit UI and review workflow
- `agent.py` - request classification, retrieval, response generation, routing
- `rag.py` - legal RAG index built on TF-IDF and FAISS
- `compliance.py` - risk and compliance heuristics
- `knowledge_base.py` - synthetic legal documents and bootstrap helpers
- `storage.py` - SQLite persistence for interactions and review state
- `legal_docs/` - local text corpus used by retrieval
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10 or newer
- pip
- Optional: OpenAI API access for LLM-generated responses

## Setup

1. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables if you want to use an LLM.

   ```bash
   export OPENAI_API_KEY="your-key"
   export OPENAI_MODEL="gpt-4o-mini"
   ```

   Optional variables:

   - `LLM_PROVIDER` - set to `openrouter` to use OpenRouter as the base URL
   - `OPENAI_BASE_URL` - custom OpenAI-compatible endpoint
   - `LEGAL_RAG_TOP_K` - number of retrieval chunks used by the RAG layer
   - `OPENAI_MAX_OUTPUT_TOKENS` - max tokens for generated responses

## Run

Start the Streamlit app:

```bash
streamlit run app.py
```

On first launch, the app will:

- create or refresh the synthetic knowledge base under `legal_docs/`,
- initialize the SQLite database `legal_agent.db`,
- build the retrieval index in memory.

## How It Works

1. You enter a legal request or upload a PDF.
2. The app classifies the request and detects sensitive areas such as bank secrecy or personal data.
3. It retrieves relevant legal chunks from the local knowledge base.
4. If `OPENAI_API_KEY` is set, the app asks the model for a formal Uzbek response.
5. The response is checked against internal compliance rules.
6. The result is routed to one of three paths:
   - `AUTO_REPLY`
   - `BANK_REVIEW`
   - `HUMAN_REVIEW_REQUIRED`
7. The full interaction is stored in SQLite and shown in the sidebar history.

## Data and Storage

- Legal source text lives in `legal_docs/*.txt`.
- Interaction history is stored in `legal_agent.db`.
- Uploaded PDF text is extracted in-memory and stored with the interaction record.

If you want a fresh start, you can delete `legal_agent.db` and rerun the app. The synthetic knowledge base will be recreated automatically if missing.

## Notes

- The app is designed for prototyping legal response workflows, not for production legal compliance.
- The generated answers should be reviewed by a qualified human before real-world use.
- If no OpenAI key is configured, the app still works using the fallback response path.

## License

No license file is currently included. Add one before distributing or reusing the project outside your own workspace.
