# agents/critic_agent.py

from crewai import Agent, Task
import json

from llms.model_loader import load_llm


def create_critic_agent(llm_seed: int | None = None):
    llm = load_llm(seed=llm_seed)
    return Agent(
        role="Molecule Critic",
        goal="Evaluate and rank validated molecules based on specifications",
        backstory="You are a medicinal chemistry expert who evaluates molecules for drug discovery and ranks them by desirability.",
        tools=[],
        verbose=True,
        llm=llm,
        allow_delegation=False
    )


def create_critic_task(validated_molecules, generation_context, original_spec, agent):
    return Task(
        description=f"""
        Evaluate and rank the following validated molecules with their generation context:

        Validated Molecules: {validated_molecules}

        Generation Context: {generation_context}

        Original Specification: {original_spec}

        Evaluate each molecule considering:
        1. Compliance with original requirements and constraints
        2. Quality of design reasoning and modifications made
        3. Drug-likeness properties (MW, logP, etc.)
        4. Synthetic accessibility and feasibility
        5. Novelty and potential for improved selectivity
        6. Expected improvements vs. actual molecular properties

        Rank molecules based on overall potential, giving higher scores to:
        - Molecules with sound design rationale
        - Better compliance with constraints
        - Optimal drug-like properties
        - Realistic synthetic accessibility
        - Novel structural features that address the requirements

        Return ONLY a compact JSON object:
        {{
          "ranked": [
            {{
              "smiles": "SMILES_STRING",
              "score": 0.85,
              "rank": 1,
              "strengths": ["strength1", "strength2"],
              "weaknesses": ["weakness1"],
              "recommendation": "Brief recommendation for this molecule"
            }},
            ...
          ],
          "summary": "Overall ranking rationale and key insights",
          "top_recommendation": "Detailed recommendation for the best candidate"
        }}
        """,
        agent=agent,
        expected_output="JSON object with ranked molecules and detailed evaluation"
    )