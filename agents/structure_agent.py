from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import SCAFFOLD_TOOLS
import json


def create_structure_agent():
    llm = load_llm()

    return Agent(
        role='Structure Analyst',
        goal='Analyze molecular scaffolds efficiently',
        backstory='Structural chemist focused on core scaffold analysis.',
        tools=SCAFFOLD_TOOLS[:3],  # Use only first 3 tools
        verbose=False,
        llm=llm,
        allow_delegation=False
    )


def create_structure_task(parsed_json: dict, agent: Agent):
    return Task(
        description=f"""
Analyze molecular structure from this spec:
{json.dumps(parsed_json, indent=2)}

TASKS:
1. **Identify core scaffold** from target molecules
2. **Find similar structures** (max 3 examples)
3. **Suggest modifications** for optimization goals

TOOL USAGE:
- Use ScaffoldAnalysisTool for core identification
- Use SimilarMoleculeFinderTool for examples
- Validate with SmilesValidatorTool

OUTPUT JSON:
{{
  "scaffold": "core_scaffold_smiles",
  "key_features": ["ring1", "functional_group1"],
  "modifications": ["suggestion1", "suggestion2"],
  "similar_molecules": [
    {{
      "smiles": "similar_smiles",
      "similarity": 0.75
    }}
  ]
}}

EFFICIENCY:
- Process first target molecule only
- Max 3 similar molecules
- Focus on actionable modifications
- Keep output concise
""",
        agent=agent,
        expected_output="Compact JSON with scaffold analysis"
    )