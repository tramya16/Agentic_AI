# agents/structure_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.chem_tools import (
    ScaffoldAnalysisTool,
    SimilarMoleculeFinderTool,
    MolecularFingerprintTool,
    SmilesValidatorTool
)
import json # Import json to format the parsed_json in the description

def create_structure_agent():
    llm = load_llm()
    scaffold_tool = ScaffoldAnalysisTool()
    similar_finder_tool = SimilarMoleculeFinderTool()
    fingerprint_tool = MolecularFingerprintTool()
    validator_tool = SmilesValidatorTool()

    return Agent(
        role='Structure Designer',
        goal='Design molecular scaffolds and core structures to meet similarity and diversity targets.',
        backstory=(
            'You are a medicinal chemist specializing in scaffold design and structure-activity relationships. '
            'You understand how to modify core structures while maintaining desired properties.'
        ),
        tools=[
            scaffold_tool,
            similar_finder_tool,
            fingerprint_tool,
            validator_tool
        ],
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

# Modified: Add an optional `context_tasks` parameter
def create_structure_task(parsed_json: dict, agent: Agent, context_tasks: list = None):
    # It's still good practice to inject the parsed_json into the description for clarity,
    # especially if the agent needs to explicitly refer to its structure.
    # The `context_tasks` will provide the full CrewAI Task object for richer context access.
    task_description = f"""
    You are a structural optimization expert. Based on the parsed JSON specification below,
    perform comprehensive scaffold analysis and propose structural modifications.

    **Parsed Specification:**
    ```json
    {json.dumps(parsed_json, indent=2)}
    ```

    Steps:
    1. Analyze the molecule's core scaffold using Murcko decomposition.
    2. Generate molecular fingerprints and perform similarity clustering.
    3. Search for structurally similar molecules for inspiration.
    4. Identify key structural features to retain.
    5. Suggest scaffold replacements or modifications that might improve the target properties (e.g. solubility, toxicity).
    6. Validate all structures with SMILES validator.

    Focus on:
    - Core scaffold identification
    - Ring system modifications
    - Relevant structural isosteres
    - Suggested replacements to maintain or improve similarity

    Return a structured output:
    - `scaffold`: canonical SMILES of core structure
    - `key_features`: list of fragments or rings to preserve
    - `modifications`: list of proposed scaffold modifications
    - `similar_molecules`: **An array of objects, where each object has "smiles" (string) and "similarity" (float). Populate this list completely from the SimilarMoleculeFinderTool output, including the actual similarity scores where available.**
    - `validation_issues`: list of any SMILES that failed validation
    """

    return Task(
        description=task_description,
        agent=agent,
        expected_output="Dictionary with scaffold SMILES, features, modifications, similar examples, and validation notes",
        # Pass the context_tasks if provided, otherwise an empty list
        context=context_tasks if context_tasks is not None else []
    )