from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import PARSER_TOOLS


def create_parser_agent():
    """Create streamlined parser agent."""
    llm = load_llm()

    return Agent(
        role="Molecular Parser",
        goal="Convert user requests into structured JSON specs",
        backstory="Expert at parsing molecular design requests into actionable specifications.",
        tools=PARSER_TOOLS,
        verbose=False,  # Reduced verbosity
        llm=llm,
        allow_delegation=False,
    )


def create_parsing_task(user_input: str, agent):
    """Create focused parsing task with minimal context."""

    description = f"""
Parse this molecular design request into valid JSON:

USER REQUEST: {user_input}

STEPS:
1. Identify target molecules (names/SMILES)
2. Extract optimization goals 
3. Identify constraints
4. Validate chemical structures

OUTPUT FORMAT (JSON only):
{{
  "target_molecules": [
    {{
      "smiles": "canonical_smiles",
      "name": "molecule_name"
    }}
  ],
  "optimization_goals": [
    {{
      "property": "logP|logS|MW|similarity",
      "direction": "maximize|minimize",
      "target": numeric_value
    }}
  ],
  "constraints": [
    {{
      "property": "MW|logP|rings",
      "min": value,
      "max": value
    }}
  ],
  "task_type": "optimization|similarity|novel_design"
}}

RULES:
- Use canonical SMILES only
- Include numeric targets where possible
- Keep constraints realistic
- Output JSON only, no explanations
"""

    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON specification"
    )