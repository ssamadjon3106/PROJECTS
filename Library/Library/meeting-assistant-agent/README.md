# 📝 Meeting Assistant Agent

An AI-powered meeting automation tool built with [Agno](https://github.com/agno-agi/agno) and [Nebius AI](https://nebius.com). Upload your meeting notes and the app automatically transcribes them, creates tasks in Linear, sends a recap to Slack, and renders a clean summary — all in one click.

---

## ✨ Features

- **Meeting Transcription** — Converts raw `.txt` meeting notes into a structured Markdown summary
- **Linear Task Creation** — Automatically generates actionable tasks with assignees and deadlines
- **Slack Notifications** — Posts a formatted recap to your `#agent-chat` channel
- **Parallel Execution** — Linear and Slack steps run simultaneously to save time
- **Streaming UI** — Real-time status updates as each workflow step completes

---

## 🏗️ Architecture

```
Upload .txt notes
       │
       ▼
┌─────────────────────────┐
│  Transcription Agent    │  Reads .txt → writes meeting_summary.md
└─────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐ ┌─────────────┐
│ Linear Agent│ │ Slack Agent │  Run in parallel
└─────────────┘ └─────────────┘
       └───────┬───────┘
               ▼
┌─────────────────────────┐
│    Summary Agent        │  Reads .txt → renders final summary
└─────────────────────────┘
```

Each agent reads directly from the uploaded `.txt` file, so every step works from the real source with no broken context chain between steps.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Framework | [Agno](https://github.com/agno-agi/agno) |
| LLM | [Kimi-K2.5](https://nebius.com) via Nebius AI |
| Frontend | [Streamlit](https://streamlit.io) |
| Task Management | [Linear](https://linear.app) |
| Notifications | [Slack](https://slack.com) |
| Async Runtime | [anyio](https://anyio.readthedocs.io) |

---

## 📋 Prerequisites

- Python 3.11+
- A [Nebius AI](https://nebius.com) account and API key
- A [Slack](https://api.slack.com/apps) bot token with `chat:write` permission in `#agent-chat`
- A [Linear](https://linear.app) API key

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ssamadjon3106/PROJECTS/meeting-assistant-agent.git
cd meeting-assistant-agent
```

### 2. Install dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or with `pip`:

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
NEBIUS_API_KEY=your_nebius_api_key
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
LINEAR_API_KEY=lin_api_your_linear_key
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Usage

1. Enter your API keys in the sidebar (or set them in `.env`)
2. Upload a `.txt` file containing your meeting notes
3. Click **▶️ Process Meeting Notes**
4. Watch the workflow run step by step
5. Read the generated summary directly in the app

### Meeting notes format

The app accepts plain `.txt` files. Conversation-style notes work best:

```
Alice: We've decided to launch on October 1st.
Bob: I'll handle the backend setup by September 20th.
Alice: Great. Bob also owns the API documentation due September 25th.
```

---

## 📁 Project Structure

```
meeting-assistant-agent/
├── app.py              # Streamlit UI and workflow runner
├── main.py             # Agno agents and workflow definition
├── pyproject.toml      # Project dependencies
├── uv.lock             # Locked dependency versions
├── .env                # API keys (not committed)
└── README.md
```

---

## ⚙️ How It Works

### Threading model
The Agno workflow runs inside a background thread with its own event loop via `anyio.run(_run, backend="asyncio")`. This avoids the `unknown async library` error that occurs when running async code inside Streamlit's runtime without an explicit backend declaration.

### Event streaming
The workflow streams events of two types:
- `RunContent` — token-by-token output from the summary agent, accumulated into the final string
- `WorkflowCompleted` — signals the end of the run and carries the complete content

### Agents
All four agents receive the uploaded file path via the workflow message and use `FileTools` to read it directly, ensuring every agent works from the real meeting content.

---

## 🔑 Slack Bot Setup

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app
2. Under **OAuth & Permissions**, add the `chat:write` scope
3. Install the app to your workspace
4. Copy the **Bot User OAuth Token** (`xoxb-…`)
5. Invite the bot to `#agent-chat`: `/invite @your-bot-name`

---

## 🔑 Linear API Key Setup

1. Go to [linear.app/settings/api](https://linear.app/settings/api)
2. Click **Create key** under Personal API Keys
3. Copy the key (`lin_api_…`)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)

---

## 🙏 Acknowledgements

- Built with [Agno](https://github.com/agno-agi/agno) — the multi-agent AI framework
- Powered by [Kimi-K2.5](https://nebius.com) on Nebius AI
- Developed with  by Samadjon Sayfullayev