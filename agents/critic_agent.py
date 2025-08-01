from crewai import Agent, Task
import json
from llms.model_loader import load_llm
from tools.tool_registry import CRITIC_TOOLS


def create_critic_agent(llm_seed: int | None = None):
    llm = load_llm(seed=llm_seed)
    return Agent(
        role="Senior Medicinal Chemistry Critic",
        goal="Provide expert evaluation and strategic optimization guidance for molecular designs",
        backstory="""You are a distinguished medicinal chemist with 15+ years in drug discovery. 
        You excel at evaluating molecular designs against complex requirements, identifying 
        structure-activity relationships, and providing actionable optimization strategies 
        for lead compound development.""",
        tools=CRITIC_TOOLS,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )


def create_critic_task(validated_molecules, generation_context, parsed_spec, agent):
    # Parse specifications
    try:
        spec_dict = json.loads(parsed_spec) if isinstance(parsed_spec, str) else parsed_spec
    except:
        spec_dict = {}

    # Extract key requirements
    design_objectives = spec_dict.get('design_objectives', [])
    structural_req = spec_dict.get('structural_requirements', {})
    bio_context = spec_dict.get('biological_context', {})
    similarity_req = spec_dict.get('similarity_requirements', {})
    success_criteria = spec_dict.get('success_criteria', [])

    return Task(
        description=f"""
Evaluate generated molecules against design requirements and provide specific feedback.

DESIGN REQUIREMENTS:
- Objectives: {', '.join(design_objectives)}
- Structural: {structural_req}
- Biological: {bio_context}
- Similarity: {similarity_req}
- Success criteria: {', '.join(success_criteria)}

MOLECULES TO EVALUATE:
Validated: {validated_molecules}
Generation context: {generation_context}

EVALUATION TASKS:
1. Calculate molecular properties
2. Assess design objective fulfillment
3. Check structural compliance
4. Evaluate drug-likeness
5. Rank molecules by overall quality
6. Provide specific optimization feedback, EXACT NUMERICAL TARGETS TO ACHIEVE OR MOVE TOWARDS.

Return JSON with this structure:
{{
  "ranked": [
    {{
      "smiles": "SMILES_STRING",
      "overall_score": 0.75,
      "rank": 1,
      "detailed_scores": {{
        "design_objective_fulfillment": 0.8,
        "structural_compliance": 0.7,
        "property_optimization": 0.8,
        "drug_likeness": 0.7
      }},
      "calculated_properties": {{"MW": 250.3, "logP": 2.1, "TPSA": 45.2}},
      "strengths": ["specific_strengths"],
      "specific_weaknesses": ["specific_issues"],
      "recommendation": "Specific structural modification with rationale"
    }}
  ],
  "generation_feedback": "Specific actionable feedback for next iteration",
  "strategic_recommendations": ["priority1", "priority2", "priority3"],
  "property_optimization_guidance": {{
    "MW": "Current X, target Y, modify by Z",
    "logP": "Current A, target B, balance with C"
  }},
  "next_iteration_priorities": ["focus_area1", "focus_area2"]
}}

CRITICAL: Provide SPECIFIC, ACTIONABLE feedback. No generic advice.
VERY VERY IMPORTANT: RETURN STRUCTURED JSON

""",
        agent=agent,
        expected_output="STRUCTURED JSON evaluation with specific optimization guidance"
    )