# --- Final Molecule Generator Agent and Task ---
from crewai import Agent, Task

from chem_tools import DrugLikenessEvaluator, MolecularPropertyCalculator, SimilarMoleculeFinderTool
from llms.model_loader import load_llm
from tools.chem_tools import (
   SmilesValidatorTool,MolecularFingerprintTool,PubChemLookupTool, MolecularSimilarityTool
)
import json # Import json to format dicts in description

def create_final_generator_agent():
    llm=load_llm()
    return Agent(
        role='Final Molecule Generator',
        goal='Synthesize inputs from all agents and reference examples to generate the final optimized molecule, perform comprehensive validation, and calculate properties.',
        backstory=(
            "You are the lead chemist in a drug discovery pipeline. Your ultimate responsibility "
            "is to integrate all the analytical insights (parsed requirements, structural modifications, "
            "property optimization strategies, and fragment-level ideas) with inspiration from "
            "existing molecules. Your task is to design a single, final, high-potential molecular "
            "candidate, ensure its chemical validity, calculate its key properties, and justify your "
            "design choices. You are meticulous and aim for the best possible outcome."
        ),
        tools=[
            SimilarMoleculeFinderTool(),
            SmilesValidatorTool(),
            DrugLikenessEvaluator(),
            MolecularFingerprintTool(),
            MolecularPropertyCalculator(),
            PubChemLookupTool(),
            MolecularSimilarityTool()
        ],
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

def create_final_generation_task(
    parsed_json: dict,
    structure_output: dict,
    property_output: dict,
    fragment_output: dict,
    agent: Agent,
    context_tasks: list
) -> Task:
    return Task(
        description=f"""
        Generate the final optimized molecule by integrating all analyses and using reference examples as inspiration.DON'T COPY REFERENCES EXACTLY, TAKE INSPIRATION.

        User's initial request:
        {json.dumps(parsed_json, indent=2)}

        Structural Analysis and Modifications from Structure Agent:
        {json.dumps(structure_output, indent=2)}

        Property Analysis and Recommendations from Property Agent:
        {json.dumps(property_output, indent=2)}

        Fragment Analysis and Bioisostere Suggestions from Fragment Agent:
        {json.dumps(fragment_output, indent=2)}

        Steps for Final Molecule Generation:
        1.  **Design Candidate Molecule**: Based on all inputs, design a novel SMILES string that:
            - Incorporates structural modifications from Structure Agent
            - Implements property optimization strategies from Property Agent
            - Uses bioisosteric replacements from Fragment Agent
            - Maintains core functionality of original target

        2.  **Validate Structure**: Use the `SmilesValidatorTool` to confirm chemical validity of your candidate

        3.  **Calculate Properties**: Use `MolecularPropertyCalculator` to get key properties:
            - Molecular Weight
            - LogP
            - H-bond donors/acceptors
            - TPSA
            - Rotatable bonds

        4.  **Evaluate Drug-likeness**: Use `DrugLikenessEvaluator` to assess:
            - Lipinski's Rule of 5 compliance
            - Ghose filter
            - Veber rules
            - PAINS alerts
            - QED score

        5.  **Check Similarity to Target**: If original target exists:
            - Use `MolecularSimilarityTool` to calculate Tanimoto similarity
            - Use 'morgan' fingerprints with threshold >0.5

        6.  **Find Reference Molecules**: Use `SimilarMoleculeFinderTool` to:
            - Find structurally similar molecules in PubChem
            - Set similarity_threshold=0.6
            - Retrieve max_results=5
            - Compare your candidate's properties with references

        7.  **PubChem Lookup**: Use `PubChemLookupTool` to:
            - Check if candidate exists in PubChem
            - Retrieve IUPAC name if available

        8.  **Finalize Design**: Optimize based on all results:
            - Improve solubility: Add polar groups (hydroxyl, amines)
            - Reduce toxicity: Remove toxicophores (nitro groups, reactive aldehydes)
            - Enhance drug-likeness: Adjust logP (target 0-5), MW (<500)

        9.  **Provide Rationale**: Explain:
            - How each agent's recommendations were implemented
            - How reference molecules influenced design
            - Property trade-offs and optimizations
            - Validation results

        **Output MUST be in the following JSON format:**
        ```json
        {{
            "final_smiles": "generated SMILES string",
            "molecule_name": "name if found in PubChem, or 'Novel compound'",
            "design_rationale": "detailed explanation of design choices",
            "property_improvements": "Specific changes made to optimize logS/LD50",
            "similarity_to_target": {{
                "target_smiles": "original target SMILES",
                "similarity_score": "Tanimoto score if calculated"
            }},
            "validation_results": {{
                "smiles_validity": "Validation output",
                "drug_likeness": "DrugLikenessEvaluator output",
                "reference_comparison": "Comparison with similar molecules"
            }},
            "final_properties": "MolecularPropertyCalculator output"
        }}
        ```
        """,
        agent=agent,
        expected_output="Comprehensive JSON object with final molecule and validation data",
        context=context_tasks
    )