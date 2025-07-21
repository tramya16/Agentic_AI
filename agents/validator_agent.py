# agents/validator_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import VALIDATOR_TOOLS
import json

def create_validator_agent(llm_seed: int):
    llm = load_llm(seed=llm_seed)
    return Agent(
        role="Molecule Validator",
        goal="Validate molecular structures, filter duplicates, and check against targets/references",
        backstory="""You are a computational chemistry expert that validates molecular structures.
        You check SMILES validity, remove duplicates (including against target molecules), 
        and apply safety filters. You are thorough but efficient.""",
        tools=VALIDATOR_TOOLS,
        verbose=True,
        llm=llm,
        allow_delegation=False,
        max_execution_time=240,  # 4 minutes max
        max_retry=2
    )


def create_validation_task(candidates: list, parsed_spec: str, agent: Agent):
    return Task(
        description=f"""
Validate {len(candidates)} candidate molecules with RELAXED criteria for learning:

Candidates: {json.dumps(candidates)}
Specification: {parsed_spec}

RELAXED VALIDATION RULES:
- Accept molecules with MW 100-600 (broader range)
- Accept logP -2 to 6 (broader range)  
- Focus on SMILES validity over strict drug-likeness
- Only flag severe toxicity concerns

Steps:
1. Check SMILES validity with smiles_validator
2. Remove exact duplicates with duplicate_check_tool
3. Apply RELAXED drug-likeness with drug_likeness_validator
4. Check severe toxicity only with toxicity_check_tool

Return EXACT JSON format:
{{
  "valid": ["list_of_valid_smiles"],
  "invalid": ["list_of_invalid_smiles"],
  "validation_details": {{
    "duplicates_found": [],
    "target_matches": [],
    "invalid_smiles": [],
    "toxicity_flags": []
  }}
}}

Be PERMISSIVE to allow learning - only reject clearly invalid molecules.
""",
        agent=agent,
        expected_output="JSON with valid/invalid molecule lists"
    )