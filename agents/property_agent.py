# agents/property_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.chem_tools import (
    DrugLikenessEvaluator, SolubilityPredictor,
    SmilesValidatorTool, MolecularFingerprintTool
)
import json # Import json to format dicts in description

def create_property_agent():
    llm = load_llm()
    return Agent(
        role='Property Optimizer',
        goal='Optimize molecular properties like logP, TPSA, QED, solubility, and toxicity.',
        backstory='You are a computational chemist expert in ADMET properties.',
        tools=[
            DrugLikenessEvaluator(),
            SolubilityPredictor(),
            SmilesValidatorTool(),
            MolecularFingerprintTool()
        ],
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

# Modified: Add an optional `context_tasks` parameter
def create_property_task(parsed_json: dict, structure_output: dict, agent: Agent, context_tasks: list = None):
    # Embed the input dictionaries into the task description for the agent
    task_description = f"""
Evaluate and optimize drug-like properties of the target molecule based on the following information:

**Parsed Task Specification:**
```json
{json.dumps(parsed_json, indent=2)}
```

**Structural Analysis Output:**
```json
{json.dumps(structure_output, indent=2)}
```

Steps:
1. Compute drug-likeness (Lipinski, Veber, QED) for the target molecule (from parsed_json or inferred from structure_output).
2. Predict aqueous solubility for the target molecule.
3. Analyze which properties deviate from the optimization goals specified in the Parsed Task Specification.
4. Suggest molecular modifications to improve properties, referencing the structural analysis.

Output structure:
{{
  "current_metrics": {{ "logP": ..., "TPSA": ..., "QED": ... }},
  "property_targets": [ ... ],
  "recommendations": [ "Add polar group", "Reduce aromatic ring count" ],
  "warnings": [],
  "validation_errors": []
}}
    """
    return Task(
        description=task_description,
        agent=agent,
        expected_output="Dictionary with metrics, targets, recommendations, and issues",
        # Pass the context_tasks if provided, otherwise an empty list
        context=context_tasks if context_tasks is not None else []
    )
