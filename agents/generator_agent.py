from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import GENERATOR_TOOLS
import json


def create_final_generator_agent():
    llm = load_llm()
    return Agent(
        role='Molecule Generator',
        goal='Generate optimized molecules using minimal but effective tool calls',
        backstory='Efficient molecular designer focused on practical results.',
        tools=GENERATOR_TOOLS,
        verbose=False,
        llm=llm,
        allow_delegation=False
    )


def create_final_generation_task(
        parsed_json: dict,
        structure_output: dict,
        property_output: dict,
        fragment_output: dict,
        agent: Agent
) -> Task:
    # Create compact context
    context = {
        "spec": parsed_json,
        "structure": structure_output,
        "property": property_output,
        "fragment": fragment_output
    }

    return Task(
        description=f"""
Generate optimized molecules based on:
{json.dumps(context, indent=2)}

WORKFLOW:
1. **Design 3-5 candidates** based on input analysis
2. **Validate each** with SmilesValidatorTool
3. **Calculate properties** with MolecularPropertyCalculator
4. **Check drug-likeness** with DrugLikenessEvaluator
5. **Find similar molecules** with SimilarMoleculeFinderTool (if needed)

TOOL USAGE OPTIMIZATION:
- Use tools sequentially, not in parallel
- Validate SMILES before property calculations
- Only use similarity tools if similarity goals exist
- Skip tools if previous ones fail

OUTPUT JSON:
{{
  "final_smiles": ["smiles1", "smiles2", "smiles3"],
  "design_rationale": "Brief explanation of approach",
  "validation_results": {{
    "valid_count": 3,
    "invalid_smiles": []
  }},
  "properties": [
    {{
      "smiles": "smiles1",
      "MW": 234.5,
      "logP": 2.1,
      "drug_like": true
    }}
  ]
}}

CONSTRAINTS:
- Generate 3-5 unique, valid SMILES
- Keep molecular weight 150-500 Da
- Ensure chemical validity
- Match user's optimization goals
- Be efficient with tool calls

FOCUS: Quality over quantity, practical molecules, efficient execution.
""",
        agent=agent,
        expected_output="JSON with final molecules and basic validation"
    )