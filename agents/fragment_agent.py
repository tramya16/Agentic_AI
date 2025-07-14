from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import FRAGMENT_TOOLS
import json

def create_fragment_agent():
    llm = load_llm()
    return Agent(
        role='Fragment Specialist',
        goal='Suggest targeted functional group modifications',
        backstory='Specialist in fragment-based molecular optimization.',
        tools=FRAGMENT_TOOLS[:2],  # Use only first 2 tools
        verbose=False,
        llm=llm,
        allow_delegation=False
    )

def create_fragment_task(parsed_json: dict, structure_output: dict, property_output: dict, agent: Agent):
    return Task(
        description=f"""
Suggest fragment modifications:

SPEC: {json.dumps(parsed_json, indent=2)}
STRUCTURE: {json.dumps(structure_output, indent=2)}
PROPERTY: {json.dumps(property_output, indent=2)}

TASKS:
1. **Identify key functional groups** in target
2. **Suggest bioisosteric replacements** for optimization
3. **Map SMARTS patterns** for modifications

TOOL USAGE:
- Use FunctionalGroupAnalyzer to identify groups
- Use BioisostereReplacer for suggestions

OUTPUT JSON:
{{
  "functional_groups": [
    {{
      "name": "carboxylic_acid",
      "smarts": "[CX3](=O)[OX2H1]",
      "position": "terminal"
    }}
  ],
  "modifications": [
    {{
      "from": "carboxylic_acid",
      "to": "tetrazole",
      "reason": "improve_solubility",
      "smarts": "[CX3](=O)[OX2H1]>>[c]1[nH][n][n][n]1"
    }}
  ],
  "priority": ["modification1", "modification2"]
}}

FOCUS:
- Target functional groups affecting optimization goals
- Provide validated SMARTS patterns
- Prioritize modifications by impact
""",
        agent=agent,
        expected_output="Targeted JSON with fragment modifications"
    )