from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import PROPERTY_TOOLS
import json

def create_property_agent():
    llm = load_llm()
    return Agent(
        role='Property Optimizer',
        goal='Analyze molecular properties efficiently',
        backstory='Computational chemist focused on key ADMET properties.',
        tools=PROPERTY_TOOLS[:3],  # Use only first 3 tools
        verbose=False,
        llm=llm,
        allow_delegation=False
    )

def create_property_task(parsed_json: dict, structure_output: dict, agent: Agent):
    return Task(
        description=f"""
Analyze properties for optimization:

SPEC: {json.dumps(parsed_json, indent=2)}
STRUCTURE: {json.dumps(structure_output, indent=2)}

ANALYSIS:
1. **Calculate current properties** of target molecule
2. **Identify gaps** vs optimization goals
3. **Suggest improvements** based on structure analysis

TOOL USAGE:
- Use MolecularPropertyCalculator for basic properties
- Use DrugLikenessEvaluator for drug-likeness
- Use SolubilityPredictor if solubility is a goal

OUTPUT JSON:
{{
  "current_properties": {{
    "MW": 234.5,
    "logP": 2.1,
    "logS": -3.2,
    "drug_like": true
  }},
  "optimization_gaps": [
    {{
      "property": "logS",
      "current": -3.2,
      "target": -2.0,
      "gap": 1.2
    }}
  ],
  "recommendations": [
    "Add polar hydroxyl group",
    "Replace benzene with pyridine"
  ]
}}

FOCUS:
- Address only properties mentioned in optimization goals
- Provide specific, actionable recommendations
- Keep calculations minimal but accurate
""",
        agent=agent,
        expected_output="Focused JSON with property analysis"
    )