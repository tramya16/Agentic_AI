# agents/validator_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import ALL_TOOLS_FLAT
import json

def create_validator_agent(llm_seed: int | None = None):
    llm = load_llm(seed=llm_seed)
    return Agent(
        role="Molecule Validator",
        goal="Validate molecular structures and filter out invalid/duplicate molecules.",
        backstory="""
You are a computational chemistry expert that validates molecular structures.

Your job is simple:
1. Check each SMILES string for validity
2. Remove duplicates (using canonical SMILES)
3. Apply basic safety filters
4. Return clean valid and invalid SMILES lists

Keep it simple and efficient.
""",
        tools=ALL_TOOLS_FLAT,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )


def create_validation_task(candidates: list, parsed_spec: str, agent: Agent):
    return Task(
        description=f"""
Validate this list of {len(candidates)} candidate molecules:

{json.dumps(candidates)}

Use the available tools to:
1. Check SMILES validity with smiles_validator
2. Remove duplicates using canonical SMILES
3. Filter out obviously problematic molecules

Return a clean JSON with two lists:
- "valid": list of valid SMILES strings
- "invalid": list of invalid SMILES strings

Example output:
{{
  "valid": ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CCO"],
  "invalid": ["invalid_smiles_here", "duplicate_smiles"]
}}

Keep it simple - just return the SMILES strings in the appropriate lists.
""",
        agent=agent,
        expected_output="JSON object with 'valid' and 'invalid' lists containing SMILES strings only."
    )