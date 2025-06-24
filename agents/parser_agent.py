from crewai import Agent
from crewai import Task

from llms.model_loader import load_llm
from tools.chem_tools import (PubChemLookupTool,SmilesValidatorTool,
                              ChemicalNameConverterTool,
                              SimilarMoleculeFinderTool)

def create_parser_agent():
    llm = load_llm()
    pubchem_tool = PubChemLookupTool()
    validator_tool = SmilesValidatorTool()
    name_convertor= ChemicalNameConverterTool()
    similar_molecules= SimilarMoleculeFinderTool()
    return Agent(
        role='Molecular Task Parser',
        goal='Convert user requests into structured molecular design tasks',
        backstory=(
            'An expert in chemical informatics that understands natural language '
            'requests about molecules and converts them to precise specifications'
        ),
        tools=[pubchem_tool,validator_tool,name_convertor,similar_molecules],
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

# def create_parsing_task(user_input: str, agent):
#     return Task(
#         description=f"""Parse the user request into structured JSON:
#         "{user_input}"
#
#         Steps:
#         1. Identify mentioned molecules and convert to canonical SMILES
#         2. Validate all SMILES strings using validator tool
#         3. Extract desired properties and constraints
#         4. For similarity requests, find reference molecules using SimilarMoleculeFinder
#         5. For property targets, find benchmark examples using PropertyBasedSearch
#         6. For commercial availability, search ZINC database
#         7. Output structured JSON with:
#            - target_molecule: Validated canonical SMILES
#            - molecule_name: IUPAC or common name
#            - desired_properties: List of properties to optimize
#            - constraints: Structural or property requirements
#            - similarity_target: Min similarity score (0-1)
#            - reference_molecules: List of similar molecules from databases
#            - property_benchmarks: Example values from reference datasets
#
#         Return ONLY valid JSON, no additional text.
#         """,
#         agent=agent,
#         expected_output="Structured JSON with validated molecular information and reference data"
#     )


def create_parsing_task(user_input: str, agent):
    return Task(
        description=f"""
You are a molecular design parser. Convert natural language requests into structured JSON specifications for molecule optimization.

--- USER REQUEST ---
{user_input}
---------------------

Follow these steps precisely:

1. **Entity Extraction**: Identify all chemical entities (names or SMILES)

2. **Name/SMILES Conversion**:
   - Convert names to canonical SMILES using `ChemicalNameConverterTool`
   - Validate all SMILES with `SmilesValidatorTool`

3. **Property Mapping**:
   - Extract desired properties and map to standard terms:
        "toxicity" → "LD50"
        "solubility" → "logS"
        "potency" → "IC50"
   - Convert qualitative terms to quantitative ranges

4. **Similarity Handling**:
   - For "similar to" requests, use `SimilarMoleculeFinderTool`
   - Set default similarity_target=0.7 if unspecified

5. **Benchmarking**:
   - For property targets, use `PropertyBasedSearchTool` to find benchmark values

6. **Commercial Search**:
   - If "purchasable" or "commercial" is mentioned, use `ZincSearchTool`

7. **Constraint Extraction**:
   - Identify structural/functional constraints
   - Extract numerical bounds where applicable

8. **Output Assembly**:
   Create VALID JSON with this structure:
   {{
     "target_molecule": {{"smiles": "CC...", "name": "..."}},
     "optimization_goals": [
       {{"property": "logS", "direction": "maximize", "target": -4}},
       {{"property": "LD50", "direction": "minimize", "target": 500}}
     ],
     "constraints": [
       {{"property": "MW", "max": 500}},
       {{"property": "similarity", "min": 0.75}}
     ],
     "reference_data": {{
       "similar_molecules": ["SMILES1", "SMILES2"],
       "property_benchmarks": [
         {{"property": "logS", "value": -3.2, "source": "TDC"}}
       ]
     }},
     "commercial_requirements": {{
       "required": true/false,
       "results": [{{"zinc_id": "...", "smiles": "..."}}]
     }}
   }}

9. **Validation**:
   - Ensure JSON is syntactically valid
   - Include only relevant fields
   - Add 'warnings' array for any assumptions made
   - Add 'validation_errors' for problematic inputs

Return ONLY the JSON object, no additional text or explanation.
DO NOT wrap the JSON in markdown code blocks (e.g., ```json ... ```).
""",
        agent=agent,
        expected_output="Valid JSON specification for molecular optimization"
    )