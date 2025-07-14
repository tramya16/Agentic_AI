from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import ALL_TOOLS


def create_validator_agent():
    llm = load_llm()

    # Use minimal essential tools
    validator_tools = (
            ALL_TOOLS.get('validation', []) +
            ALL_TOOLS.get('descriptors', [])[:2] +  # Only first 2 descriptor tools
            ALL_TOOLS.get('drug_likeness_and_safety', [])[:2]  # Only first 2 safety tools
    )

    return Agent(
        role='Molecular Validator',
        goal='Efficiently validate molecules with essential checks only',
        backstory='Quality control specialist focused on critical validation metrics.',
        tools=validator_tools,
        verbose=False,
        llm=llm,
        allow_delegation=False
    )


def create_validation_task(candidates: list, parsed_spec: dict, agent):
    return Task(
        description=f"""
Validate these molecules efficiently:

CANDIDATES: {candidates[:10]}  # Limit to first 10
SPEC: {parsed_spec}

STREAMLINED VALIDATION:
1. **SMILES Validation** - Check each with SmilesValidatorTool
2. **Basic Properties** - Calculate MW, logP, TPSA
3. **Drug-likeness** - Quick Lipinski check
4. **Constraint Check** - Verify spec requirements

SCORING (0-1 scale):
- structural_valid: SMILES validity
- property_score: Meets property targets
- overall_score: Average of above

ERROR HANDLING:
- Skip failed tools, continue validation
- Use fallback scoring if tools fail
- Limit tool calls to essential ones

OUTPUT JSON:
{{
  "validated_candidates": [
    {{
      "smiles": "canonical_smiles",
      "valid": true,
      "scores": {{
        "structural_valid": 1.0,
        "property_score": 0.8,
        "overall_score": 0.9
      }},
      "properties": {{
        "MW": 234.5,
        "logP": 2.1
      }},
      "issues": []
    }}
  ],
  "summary": {{
    "total": 5,
    "valid": 4,
    "avg_score": 0.75
  }}
}}

EFFICIENCY RULES:
- Process max 10 candidates
- Use 3-5 tools maximum per molecule
- Fail fast on invalid SMILES
- Provide concise output
""",
        agent=agent,
        expected_output='Efficient JSON validation report'
    )