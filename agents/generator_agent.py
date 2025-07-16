# agents/generator_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import ALL_TOOLS_FLAT
import json


def create_generator_agent(llm_seed: int | None = None):
    llm = load_llm(seed=llm_seed)
    return Agent(
        role="Molecule Generator",
        goal="Generate novel molecular structures with detailed reasoning based on specifications",
        backstory="""You are a computational chemist expert at designing new molecules using similarity search 
        and structure modification. You provide detailed reasoning for each design decision and focus on quality 
        over quantity.""",
        tools=ALL_TOOLS_FLAT,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )


def create_generation_task(parsed_spec: str, agent: Agent):
    return Task(
        description=f"""
        Generate candidate molecules based on the following specification:

        Specification: {parsed_spec}

        Use the available tools to:
        1. Analyze the target_molecules and understand their structure-activity relationships
        2. Identify key functional groups and structural features
        3. Generate 3 diverse candidate molecules with strategic modifications 
        4. For each candidate, provide detailed reasoning

        IMPORTANT GUIDELINES:
        - KEEP LIST OF TARGET AND REFERENCE SMILES, MAKE SURE GENERATED SMILES IS NOT A COPY.
        - DO NOT COPY target/reference molecules exactly - create meaningful variations
        - Focus on quality over quantity - 3 well-reasoned candidates
        - Each modification should address the specified constraints/properties
        - Consider drug-likeness, synthetic accessibility, and selectivity

        Return ONLY a compact JSON object with this structure:
        {{
          "candidates": [
            {{
              "smiles": "SMILES_STRING",
              "reasoning": "Detailed explanation of design rationale, modifications made, and expected properties",
              "modifications": ["modification1", "modification2"],
              "expected_improvements": ["improvement1", "improvement2"]
            }},
            ...
          ],
          "design_strategy": "Overall approach and methodology used",
          "target_analysis": "Analysis of target molecule(s) and key insights"
        }}
        """,
        agent=agent,
        expected_output="JSON object with candidate molecules, reasoning, and design strategy"
    )