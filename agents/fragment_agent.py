# agents/fragment_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.chem_tools import (
    FunctionalGroupManager, BioisostereGenerator, SmilesValidatorTool
)
import json # Import json to format dicts in description

def create_fragment_agent():
    llm = load_llm()
    return Agent(
        role='Fragment Specialist',
        goal='Modify functional groups using SMARTS to optimize properties.',
        backstory='Synthetic chemist focused on fragment-based molecular design.',
        tools=[
            FunctionalGroupManager(),
            BioisostereGenerator(),
            SmilesValidatorTool()
        ],
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

# Modified: Add an optional `context_tasks` parameter
def create_fragment_task(parsed_json: dict, structure_output: dict, property_output: dict, agent: Agent, context_tasks: list = None):
    # Embed all relevant input dictionaries into the task description
    task_description = f"""
Analyze functional groups and make replacement/modification recommendations based on the following:

**Parsed Task Specification:**
```json
{json.dumps(parsed_json, indent=2)}
```

**Structural Analysis Output:**
```json
{json.dumps(structure_output, indent=2)}
```

**Property Optimization Output:**
```json
{json.dumps(property_output, indent=2)}
```

Steps:
1. Map functional groups for the target molecule using SMARTS.
2. Assess their impact on key properties, considering the Property Optimization Output.
3. Propose fragment additions/removals to achieve optimization goals.
4. Suggest bioisosteric replacements for functional groups identified, especially if they are problematic or need improvement.

Expected format:
{{
  "functional_groups": [...],
  "modifications": [...],
  "bioisosteres": [...],
  "smarts_patterns": [...],
  "validation_issues": []
}}
    """
    return Task(
        description=task_description,
        agent=agent,
        expected_output="Fragment analysis with functional map and SMARTS-based proposals",
        # Pass the context_tasks if provided, otherwise an empty list
        context=context_tasks if context_tasks is not None else []
    )
