from agno.agent import Agent
from agno.models.nebius import Nebius
from src.config import EVAL_MODEL  
from src.prompt_loader import load_prompts 


def create_evaluator(api_key: str, candidates: list) -> Agent:
    prompts = load_prompts()


    desc = prompts["single_description"]
    instr = prompts["single_instructions"]

    return Agent(
        model=Nebius(id=EVAL_MODEL, api_key=api_key),
        description=desc,
        instructions=instr,
        markdown=False  # disable markdown so JSON output stays clean
    )