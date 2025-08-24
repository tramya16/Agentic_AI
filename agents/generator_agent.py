from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import GENERATOR_TOOLS
import json


def create_generator_agent(llm_seed: int, model_name="gemini_2.0_flash"):
    llm = load_llm(seed=llm_seed,model_name=model_name)
    return Agent(
        role="Molecular Designer",
        goal="Generate novel molecular structures using medicinal chemistry principles and iterative optimization",
        backstory="""You are an expert medicinal chemist with deep knowledge of structure-activity 
        relationships, pharmacophore modeling, and lead optimization. You apply systematic design 
        strategies, incorporate SAR feedback, and ensure chemical feasibility in your molecular designs.""",
        tools=GENERATOR_TOOLS,
        verbose=True,
        llm=llm,
        allow_delegation=False,
        max_execution_time=300,
        max_retry=2
    )


def create_generation_task(parsed_spec: str, agent: Agent, critic_feedback: str = None,
                           generation_history: list = None, iteration_number: int = 1):
    # Parse specification
    try:
        spec_dict = json.loads(parsed_spec) if isinstance(parsed_spec, str) else parsed_spec
    except:
        spec_dict = {}

    # Build feedback section with structure
    feedback_section = ""
    if critic_feedback:
        feedback_section = f"""
OPTIMIZATION FEEDBACK FROM PREVIOUS ITERATION:
{critic_feedback[:400]}

REQUIREMENT: Address each feedback point with specific structural modifications.
"""

    # Structure learning from history
    history_section = ""
    if generation_history and len(generation_history) > 0:
        recent_history = generation_history[-1] if generation_history else {}
        previous_smiles = recent_history.get('generated_smiles', [])

        if previous_smiles:
            history_section = f"""
DESIGN EVOLUTION CONSTRAINTS:
- Avoid direct repetition: {', '.join(previous_smiles[:5])}
- Build upon successful features from previous iterations
"""

    # Extract design context with more structure
    objectives = spec_dict.get('design_objectives', [])
    structural_req = spec_dict.get('structural_requirements', {})
    reference_molecules = spec_dict.get('similarity_requirements', {}).get('reference_molecules', [])

    return Task(
        description=f"""
Design {5} novel molecular candidates for iteration {iteration_number} using systematic medicinal chemistry approaches.
VERY VERY IMPORTANT: RETURN STRUCTURED JSON RESPONSE

DESIGN OBJECTIVES: {', '.join(objectives[:3])}
REFERENCE STRUCTURE: {reference_molecules[0] if reference_molecules else 'None'}
CORE SCAFFOLD: {structural_req.get('core_scaffold', 'Flexible')}
ESSENTIAL FEATURES: {', '.join(structural_req.get('essential_groups', []))}

{feedback_section}
{history_section}

IMPORTANT: DON NOT GENERATE IDENTICAL TO REFERENCE STRUCTURE
DESIGN PRINCIPLES:
1. Generate chemically valid SMILES structures
2. Maintain pharmacophore integrity while introducing modifications
3. Apply structure-activity relationship principles
4. Ensure synthetic feasibility and drug-like properties
5. Demonstrate clear design rationale for each modification

Return JSON with this structure:
{{
  "candidates": [
    {{
      "smiles": "VALID_SMILES_STRING",
      "design_rationale": "Scientific justification for design choices",
      "modifications_made": ["specific_structural_change1", "specific_structural_change2"],
      "expected_improvements": ["predicted_benefit1", "predicted_benefit2"],
      "sar_rationale": "Structure-activity relationship reasoning"
    }}
  ],
  "design_strategy": "Overall medicinal chemistry approach for this iteration",
  "pharmacophore_analysis": "Key pharmacophore elements maintained or modified"
}}

VALIDATION REQUIREMENTS:
- All SMILES must be RDKit-parseable
- Maintain similarity to reference while introducing novelty
- Address specific feedback from previous iterations
- Demonstrate systematic design thinking

VERY VERY IMPORTANT: RETURN STRUCTURED JSON
""",
        agent=agent,
        expected_output="STRUCTURED JSON with molecular designs and comprehensive rationale"
    )
