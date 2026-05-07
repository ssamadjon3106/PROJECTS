"""
Meeting Assistant Agent — Workflow Definition
All agents read directly from the uploaded .txt meeting notes file.
"""

import os

from agno.agent import Agent
from agno.models.nebius import Nebius
from agno.tools.file import FileTools
from agno.tools.linear import LinearTools
from agno.tools.slack import SlackTools
from agno.workflow import Step, Workflow
from agno.workflow.parallel import Parallel
from dotenv import load_dotenv

load_dotenv()



def get_model() -> Nebius:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise EnvironmentError("NEBIUS_API_KEY is not set.")
    return Nebius(id="moonshotai/Kimi-K2.5", api_key=api_key)




def get_slack_tools() -> SlackTools:
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        raise EnvironmentError("SLACK_BOT_TOKEN is not set.")
    return SlackTools(token=token)


def get_linear_tools() -> LinearTools:
    api_key = os.getenv("LINEAR_API_KEY")
    if not api_key:
        raise EnvironmentError("LINEAR_API_KEY is not set.")
    return LinearTools(api_key=api_key)



def build_transcription_agent(model: Nebius) -> Agent:
    return Agent(
        name="Meeting Transcription Agent",
        model=model,
        tools=[FileTools()],
        instructions=(
            "You are a meeting transcription assistant. "
            "The user message contains the path to a .txt file with raw meeting notes. "
            "Read that file first (do NOT modify it). "
            "Then write a detailed Markdown summary to `./meeting_summary.md` "
            "with the following sections:\n\n"
            "- **Project Overview**: Goals and scope.\n"
            "- **Cost Estimates**: Every figure mentioned.\n"
            "- **Product Tiers**: Features per tier.\n"
            "- **Technical Stack**: Technologies agreed upon.\n"
            "- **Timeline**: Deadlines and milestones.\n"
            "- **Decisions Made**: Key agreements reached.\n"
            "- **Action Items**: Every task with owner name and deadline.\n\n"
            "Use only real content from the notes. "
            "Do not invent or use placeholder text."
        ),
        markdown=True,
    )


def build_linear_agent(model: Nebius, linear_tools: LinearTools) -> Agent:
    return Agent(
        name="Linear Task Agent",
        model=model,
        tools=[FileTools(), linear_tools],
        instructions=(
            "You are a productivity assistant. "
            "The user message contains the path to a .txt file with raw meeting notes. "
            "Read that file first. "
            "Then create clear, actionable tasks in Linear based on its content. "
            "For every action item include:\n"
            "- A concise title.\n"
            "- A detailed description referencing the meeting discussion.\n"
            "- The assignee's name.\n"
            "- The deadline, if mentioned.\n"
            "- A priority level if implied by the discussion.\n\n"
            "Use only real content from the notes. "
            "Do not invent or use placeholder text."
        ),
        markdown=True,
    )


def build_slack_agent(model: Nebius, slack_tools: SlackTools) -> Agent:
    return Agent(
        name="Slack Notification Agent",
        model=model,
        tools=[FileTools(), slack_tools],
        instructions=(
            "You are a communication assistant. "
            "The user message contains the path to a .txt file with raw meeting notes. "
            "Read that file first. "
            "Then send a concise, friendly recap to the #agent-chat Slack channel "
            "using this structure:\n\n"
            "🎯 *Key Decisions*\n"
            "• One decision per bullet.\n\n"
            "📋 *Assigned Tasks*\n"
            "• Task — Owner — Deadline\n\n"
            "🚀 *Next Steps*\n"
            "• One next step per bullet.\n\n"
            "End with an encouraging closing line and mention that tasks "
            "have been created in Linear. "
            "Keep every bullet to one line. "
            "Use only real content from the notes. "
            "Do not invent or use placeholder text."
        ),
    )


def build_summary_agent(model: Nebius) -> Agent:
    return Agent(
        name="Meeting Summary Agent",
        model=model,
        tools=[FileTools()],
        instructions=(
            "You are a summarisation assistant. "
            "The user message contains the path to a .txt file with raw meeting notes. "
            "Read that file first. "
            "Then generate a polished Markdown summary using this exact structure:\n\n"
            "# 📋 Meeting Summary\n\n"
            "## 🎯 Main Topics\n"
            "- List every key discussion topic.\n\n"
            "## 💡 Key Decisions\n"
            "| Decision | Details |\n"
            "|----------|---------|\n"
            "| <real decision> | <real details> |\n\n"
            "## 📝 Action Items\n"
            "| Task | Owner | Deadline |\n"
            "|------|-------|----------|\n"
            "| <real task> | <real owner> | <real deadline> |\n\n"
            "## 🚀 Next Steps\n"
            "- List every concrete next step.\n\n"
            "_Tasks have been created in Linear and a summary posted to Slack._\n\n"
            "Use only real content from the notes. "
            "Do not invent or use placeholder text."
        ),
        markdown=True,
    )


def build_workflow() -> Workflow:
    model = get_model()
    slack_tools = get_slack_tools()
    linear_tools = get_linear_tools()

    return Workflow(
        name="Meeting Assistant Workflow",
        steps=[
            Step(
                name="Meeting Transcription Task",
                agent=build_transcription_agent(model),
            ),
            Parallel(
                Step(
                    name="Linear Task",
                    agent=build_linear_agent(model, linear_tools),
                ),
                Step(
                    name="Slack Notification Task",
                    agent=build_slack_agent(model, slack_tools),
                ),
                name="Notification Tasks",
            ),
            Step(
                name="Summary Task",
                agent=build_summary_agent(model),
            ),
        ],
    )



workflow = build_workflow()


if __name__ == "__main__":
    workflow.print_response(
        "Process the meeting notes from ./meeting_notes.txt: "
        "summarise, create Linear tasks, and send a Slack notification."
    )