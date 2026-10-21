import yaml
from pathlib import Path
from typing import TypedDict

PROMPT_PATH = Path(__file__).parent / "prompts.yaml"


class PromptConfig(TypedDict):
    single_description: str
    single_instructions: str


def load_prompts() -> PromptConfig:
    with PROMPT_PATH.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict) or not all(
        isinstance(value.get(key), str)
        for key in ("single_description", "single_instructions")
    ):
        raise ValueError("Prompt configuration is incomplete.")
    return value
