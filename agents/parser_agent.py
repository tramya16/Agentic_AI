from crewai import Agent
from crewai import Task

from llms.gemini_loader import load_gemini_llm
from tools.chem_tools import PubChemLookupTool,Smilesvalidatortool

def create_parser_agent():
    llm = load_gemini_llm()
    pubchem_tool = PubChemLookupTool()
    validator_tool = Smilesvalidatortool()

    return Agent(
        role='Molecular Task Parser',
        goal='Convert user requests into structured molecular design tasks',
        backstory=(
            'An expert in chemical informatics that understands natural language '
            'requests about molecules and converts them to precise specifications'
        ),
        tools=[pubchem_tool,validator_tool],
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

def create_parsing_task(user_input: str, agent):
    return Task(
        description=f"""Parse the user request into structured JSON:
        "{user_input}"

        Steps:
        1. Identify mentioned molecules and convert to SMILES
        2. Validate all SMILES strings
        3. Extract desired properties and constraints
        4. Output structured JSON with:
           - target_molecule: Validated SMILES
           - molecule_name: Common/IUPAC name
           - desired_properties: List of properties
           - constraints: Any specific requirements
           - similarity_target: Desired similarity range

        Return ONLY valid JSON, no additional text.
        """,
        agent=agent,
        expected_output="Structured JSON with validated molecular information"
    )