import yaml
from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "prompts.yaml"


def load_prompts():
    with open(PROMPT_PATH, "r") as f:
        return yaml.safe_load(f)