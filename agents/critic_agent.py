# agents/critic_agent.py

from crewai import Agent, Task
import json
from llms.model_loader import load_llm
from tools.tool_registry import CRITIC_TOOLS


def create_critic_agent(llm_seed: int | None = None):
    llm = load_llm(seed=llm_seed)
    return Agent(
        role="Molecule Critic",
        goal="Evaluate validated molecules and provide actionable feedback for iterative improvement",
        backstory="""You are a medicinal chemistry expert who evaluates molecules and provides actionable feedback 
        for iterative improvement. You identify specific weaknesses and provide concrete suggestions for the next 
        iteration of molecular design.""",
        tools=CRITIC_TOOLS,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )


def create_critic_task(validated_molecules, generation_context, parsed_spec, agent):
    return Task(
        description=f"""
Evaluate molecules and provide ACTIONABLE feedback:

Validated: {validated_molecules}
Context: {generation_context}
Spec: {parsed_spec}

Use tools to calculate properties and provide specific, actionable feedback.

Return EXACT JSON format:
{{
  "ranked": [
    {{
      "smiles": "SMILES_STRING",
      "score": 0.75,
      "rank": 1,
      "properties": {{"MW": 250.3, "logP": 2.1}},
      "strengths": ["good_drug_likeness", "novel_scaffold"],
      "weaknesses": ["low_target_similarity"],
      "recommendation": "Specific improvement suggestion"
    }}
  ],
  "summary": "Brief overall assessment",
  "top_recommendation": "Best molecule and why",
  "generation_feedback": "SPECIFIC actionable feedback: Increase similarity to target by adding hydroxyl groups. Reduce molecular weight by replacing tert-butyl with methyl. Improve selectivity by modifying aromatic substitution pattern.",
  "key_improvements_needed": ["improve_target_similarity", "reduce_MW"],
  "specific_suggestions": [
    "Add hydroxyl group at position X",
    "Replace bulky substituent Y with smaller group Z",
    "Modify ring system for better binding"
  ]
}}

Make feedback SPECIFIC and ACTIONABLE for next iteration.
""",
        agent=agent,
        expected_output="JSON with ranked molecules and specific improvement feedback"
    )