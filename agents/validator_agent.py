from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import VALIDATOR_TOOLS
import json

def create_validator_agent(llm_seed: int):
    llm = load_llm(seed=llm_seed)
    return Agent(
        role="Molecular Validation Specialist",
        goal="Apply comprehensive validation filters to ensure molecular quality and compliance",
        backstory="""You are a computational chemistry expert specializing in molecular validation, 
        ADMET prediction, and drug-likeness assessment. You apply systematic filters to ensure 
        generated molecules meet quality standards for drug discovery.""",
        tools=VALIDATOR_TOOLS,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )


def create_validation_task(candidates: list, parsed_spec: str, agent: Agent):
    return Task(
        description=f"""
Validate {len(candidates)} candidate molecules:

Candidates: {json.dumps(candidates)}
Specification: {parsed_spec}

Validation steps:
1. Check SMILES validity
2. Remove duplicates (including against reference molecules)
3. Check basic drug-likeness (relaxed criteria)
4. Flag severe toxicity concerns only

Return JSON:
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

Use relaxed criteria - only reject clearly invalid molecules.
""",
        agent=agent,
        expected_output="JSON with valid/invalid molecule classification"
    )