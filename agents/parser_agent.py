from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import PARSER_TOOLS


def create_parser_agent(model_config=None, llm_seed: int | None = None):
    llm = load_llm(model_config=model_config, seed=llm_seed)
    return Agent(
        role="Molecular Query Analyst",
        goal="Parse molecular design queries into comprehensive, actionable specifications for drug discovery",
        backstory="""You are a senior computational medicinal chemist with expertise in structure-based 
        drug design. You excel at translating complex molecular design requests into structured 
        specifications that capture pharmacophore requirements, SAR insights, and optimization objectives.""",
        tools=PARSER_TOOLS,
        verbose=False,
        llm=llm,
        allow_delegation=False
    )


def create_parsing_task(user_input: str, agent: Agent):
    return Task(
        description=f"""
Parse this molecular design query into structured JSON specification:

USER QUERY: {user_input}

ANALYSIS REQUIREMENTS:
1. Identify target molecules (SMILES or names) and extract key structural information
2. Determine primary design objectives and secondary goals
3. Extract structural requirements including core scaffolds and essential groups
4. Identify property constraints and optimization targets
5. Understand biological context and therapeutic goals
6. Define similarity requirements and reference molecules

OUTPUT STRUCTURE - Return valid JSON:
{{
  "target_molecules": ["SMILES_or_names"],
  "task_type": "similarity_search|optimization|scaffold_hopping|isomers",
  "design_objectives": ["primary_goal", "secondary_goals"],
  "structural_requirements": {{
    "core_scaffold": "description_or_SMARTS",
    "essential_groups": ["functional_group1", "functional_group2"],
    "pharmacophore": ["pharmacophore_feature1", "pharmacophore_feature2"],
    "forbidden_groups": ["avoid_group1", "avoid_group2"]
  }},
  "properties": {{
    "MW": "range_or_target_value",
    "logP": "range_or_target_value", 
    "TPSA": "range_or_target_value",
    "HBD": "range_or_target_value",
    "HBA": "range_or_target_value"
  }},
  "biological_context": {{
    "target_protein": "protein_name_or_class",
    "activity_type": "binding|inhibition|agonism|antagonism",
    "therapeutic_area": "indication_or_disease"
  }},
  "similarity_requirements": {{
    "reference_molecules": ["reference_SMILES"],
    "similarity_threshold": 0.6,
    "similarity_type": "tanimoto|morgan|atom_pair"
  }},
  "constraints": ["drug_like_properties", "specific_constraints"],
  "modification_strategy": ["specific_structural_changes_requested"],
  "design_rationale": "context_and_scientific_reasoning",
  "success_criteria": ["evaluation_metrics", "performance_targets"]
}}

CRITICAL REQUIREMENTS:
- Extract all relevant structural and property information
- Identify the specific task type accurately
- Preserve scientific context and rationale
- Ensure all SMILES are valid if provided
- Return properly formatted JSON only

VALIDATION: Ensure all extracted information is scientifically accurate and the JSON is valid.
""",
        agent=agent,
        expected_output="Valid JSON specification for molecular design with comprehensive requirements"
    )
