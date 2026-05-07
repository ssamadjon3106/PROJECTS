"""
Meeting Assistant Agent — Streamlit UI
Runs the agno workflow in a dedicated thread so anyio can detect the asyncio
backend correctly (no nest_asyncio required).
"""

import os
import queue
import threading
from typing import Optional

import anyio
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(
    page_title="Meeting Assistant Agent",
    page_icon="📝",
    layout="wide",
)


@st.cache_resource
def get_workflow():
    from main import build_workflow  
    return build_workflow()




def run_workflow(file_path: str, status_widget) -> Optional[str]:
    """
    Executes the agno workflow in a background thread with its own event loop.
    - RunContent tokens are accumulated to reconstruct the full streamed output.
    - WorkflowCompleted carries the final content; falls back to accumulated tokens.
    Status updates are passed back to the main thread via a queue.
    Returns the final summary string, or None on failure.
    """
    status_queue: queue.Queue = queue.Queue()
    result: dict = {"content": None, "error": None}

    async def _run():
        accumulated: list[str] = []
        wf = get_workflow()
        try:
            async for event in wf.arun(
                message=(
                    f"Process the meeting notes from {file_path}: "
                    "summarise, create Linear tasks, and send a Slack notification."
                ),
                markdown=True,
                stream=True,
                stream_intermediate_steps=True,
            ):
                status_queue.put(("event", event))

                
                event_name = (
                    event.event
                    if isinstance(event.event, str)
                    else event.event.value
                )

               
                if event_name == "RunContent":
                    token = getattr(event, "content", "") or ""
                    accumulated.append(token)

                
                elif event_name in ("WorkflowCompleted", "workflow_completed"):
                    completed_content = getattr(event, "content", "") or ""
                    result["content"] = completed_content or "".join(accumulated)

        except Exception as exc:
            print(f"[WORKFLOW ERROR] {exc}")
            result["error"] = exc
        finally:
            
            if not result["content"] and accumulated:
                result["content"] = "".join(accumulated)
            status_queue.put(("done", None))

    def _thread():
        anyio.run(_run, backend="asyncio")

    thread = threading.Thread(target=_thread, daemon=True)
    thread.start()

   

    STEP_ICONS = {
        "WorkflowStarted":              "▶️",
        "WorkflowCompleted":            "🏁",
        "workflow_completed":           "🏁",
        "StepStarted":                  "🚀",
        "StepCompleted":                "✅",
        "ParallelExecutionStarted":     "🔄",
        "ParallelExecutionCompleted":   "✅",
        "RunContent":                   "✍️",
    }

    while True:
        try:
            msg_type, payload = status_queue.get(timeout=0.3)
        except queue.Empty:
            continue

        if msg_type == "done":
            break

        event = payload
        event_name = (
            event.event if isinstance(event.event, str) else event.event.value
        )

       
        if event_name == "RunContent":
            continue

        step_name = getattr(event, "step_name", "")
        icon = STEP_ICONS.get(event_name, "⚙️")
        label = f"{icon} {event_name.replace('_', ' ').title()}"
        if step_name:
            label += f": {step_name}"
        status_widget.update(label=label)

    thread.join()

    if result["error"]:
        raise result["error"]
    return result["content"]


with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    nebius_key = st.text_input(
        "Nebius API Key",
        value=os.getenv("NEBIUS_API_KEY", ""),
        type="password",
        placeholder="nbs-…",
    )
    slack_token = st.text_input(
        "Slack Bot Token",
        value=os.getenv("SLACK_BOT_TOKEN", ""),
        type="password",
        placeholder="xoxb-…",
    )
    linear_key = st.text_input(
        "Linear API Key",
        value=os.getenv("LINEAR_API_KEY", ""),
        type="password",
        placeholder="lin_api_…",
    )

    if st.button("💾 Save Keys", use_container_width=True):
        if nebius_key:
            os.environ["NEBIUS_API_KEY"] = nebius_key
        if slack_token:
            os.environ["SLACK_BOT_TOKEN"] = slack_token
        if linear_key:
            os.environ["LINEAR_API_KEY"] = linear_key
        st.success("Keys saved for this session.")

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📄 Upload Meeting Notes",
        type=["txt"],
        accept_multiple_files=False,
    )

    if uploaded_file:
        save_path = f"./{uploaded_file.name}"
        with open(save_path, "wb") as fh:
            fh.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: `{uploaded_file.name}`")

    process_btn = st.button(
        "▶️ Process Meeting Notes",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    st.markdown("---")
    st.caption(
        "Developed with by "
        "Samadjon Sayfullayev"
    )


st.title("📝 Meeting Assistant Agent")
st.markdown(
    "**Streamline your meetings with AI-powered transcription, "
    "task creation, and notifications.**"
)
st.divider()

ABOUT_MD = """
## What this app does

| Step | Agent | Output |
|------|-------|--------|
| 1 | **Transcription Agent** | Writes `meeting_summary.md` from your notes |
| 2a | **Linear Agent** | Creates actionable tasks in Linear |
| 2b | **Slack Agent** | Posts a recap to `#agent-chat` |
| 3 | **Summary Agent** | Renders the final summary here |

Steps 2a and 2b run in **parallel** to save time.

### Getting started
1. Enter your API keys in the sidebar.
2. Upload a `.txt` file with your meeting notes.
3. Click **Process Meeting Notes**.
"""

if process_btn:
    if not uploaded_file:
        st.warning("Please upload meeting notes first.")
    else:
        try:
            with st.status("Starting workflow…", expanded=True) as status:
                summary = run_workflow(f"./{uploaded_file.name}", status)
                status.update(label="✅ Processing complete!", state="complete")

            if summary:
                st.markdown(summary)
            else:
                st.warning(
                    "The workflow finished but returned no summary. "
                    "Check the terminal for `[EVENT]` lines to diagnose further."
                )

        except EnvironmentError as env_err:
            st.error(f"Missing API key — {env_err}")
        except Exception as err:
            st.error(f"Workflow error: {err}")

else:
    st.markdown(ABOUT_MD)