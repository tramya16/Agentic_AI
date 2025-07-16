# agents/parser_agent.py

from crewai import Agent, Task

from llms.model_loader import load_llm
from tools.tool_registry import PARSER_TOOLS
import json


def create_parser_agent(llm_seed: int | None = None):
    llm = load_llm(seed=llm_seed)
    return Agent(
        role="Query Parser",
        goal="Parse user queries into structured specifications for molecular design",
        backstory="You are an expert in understanding molecular design requests and converting them into structured data.",
        tools=PARSER_TOOLS,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )


def create_parsing_task(user_input: str, agent: Agent):
    return Task(
        description=f"""
        Parse the following user query into a structured JSON specification:

        User Query: {user_input}

        Extract and structure the following information:
        - target_molecules: List of SMILES strings or molecule names mentioned
        - properties: Dictionary of desired properties (e.g., logP, MW, solubility)
        - constraints: List of constraints or requirements
        - task_type: One of ["generation", "optimization", "analysis"]
        - similarity_threshold: Float between 0-1 if similarity is mentioned

        Return ONLY a compact JSON object with the parsed specification.
        Example output:
        {{"target_molecules": ["CCO"], "properties": {{"MW": "<300"}}, "constraints": ["drug-like"], "task_type": "generation"}}
        """,
        agent=agent,
        expected_output="A compact JSON object containing the parsed specification"
    )