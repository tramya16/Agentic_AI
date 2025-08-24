from crewai import Agent, Task

from llms.model_loader import load_llm
from tools.tool_registry import PARSER_TOOLS


def create_parser_agent(llm_seed: int | None = None, model_name="gemini_2.0_flash"):
    llm = load_llm(seed=llm_seed,model_name=model_name)
    return Agent(
        role="Molecular Query Analyst",
        goal="Parse molecular design queries into comprehensive, actionable specifications for drug discovery",
        backstory="""You are a senior computational medicinal chemist with expertise in structure-based 
        drug design. You excel at translating complex molecular design requests into structured 
        specifications that capture pharmacophore requirements, SAR insights, and optimization objectives.""",
        tools=PARSER_TOOLS,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )


def create_parsing_task(user_input: str, agent: Agent):
    return Task(
        description=f"""
        Parse this molecular design query into structured JSON:
        User Query: {user_input}

        Extract key information:
        1. Target molecules (SMILES or names)
        2. Design objectives
        3. Structural requirements
        4. Property constraints
        5. Biological context
        6. Similarity requirements

        Return JSON with this structure:
        {{
          "target_molecules": ["SMILES_or_names"],
          "task_type": "similarity_search|optimization|scaffold_hopping",
          "design_objectives": ["primary_goal", "secondary_goals"],
          "structural_requirements": {{
            "core_scaffold": "description",
            "essential_groups": ["group1", "group2"],
            "pharmacophore": ["feature1", "feature2"]
          }},
          "properties": {{
            "MW": "range_or_target",
            "logP": "range_or_target",
            "TPSA": "range_or_target"
          }},
          "biological_context": {{
            "target_protein": "protein_name",
            "activity_type": "binding|inhibition|agonism",
            "therapeutic_area": "indication"
          }},
          "similarity_requirements": {{
            "reference_molecules": ["SMILES"],
            "similarity_threshold": 0.6,
            "similarity_type": "tanimoto"
          }},
          "constraints": ["drug_like", "specific_constraints"],
          "modification_strategy": ["specific_changes_requested"],
          "design_rationale": "context_and_reasoning",
          "success_criteria": ["evaluation_metrics"]
        }}
        
       VERY VERY IMPORTANT: RETURN STRUCTURED JSON  
        """,
        agent=agent,
        expected_output="STRUCTURED JSON specification for molecular design"
    )
