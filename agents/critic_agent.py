# agents/critic_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm

def create_critic_agent():
    llm = load_llm()
    return Agent(
        role='Molecular Design Critic',
        goal=(
            'Evaluate validated molecular candidates and decide whether to accept them, '
            'request refinements, or restart the generation process.'
        ),
        backstory=(
            'You are a senior medicinal chemist with decades of experience in drug discovery. '
            'You have a keen eye for identifying promising molecular candidates and can spot '
            'potential issues that automated tools might miss. Your expertise guides the '
            'iterative refinement process.'
        ),
        tools=[],
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

def create_critic_task(validated_candidates: list, parsed_spec: dict, iteration: int, agent):
    return Task(
        description=f"""
You are the Molecular Design Critic. Evaluate these validated candidates and decide the next action:

VALIDATED CANDIDATES:
{validated_candidates}

ORIGINAL SPECIFICATION:
{parsed_spec}

ITERATION: {iteration} (max 3 iterations)

EVALUATION CRITERIA:
1. QUALITY ASSESSMENT:
   - Overall scores (target: >0.7 for acceptance)
   - Property target achievement
   - Safety profile adequacy
   - Structural diversity in top candidates

2. SPECIFICATION ADHERENCE:
   - Meets all hard constraints
   - Achieves optimization goals
   - Satisfies similarity requirements
   - Avoids forbidden substructures

3. PRACTICAL CONSIDERATIONS:
   - Synthetic accessibility
   - Drug-likeness for pharmaceutical applications
   - Novelty vs. precedent balance
   - Potential for further optimization

DECISION LOGIC:
- ACCEPT: ≥5 candidates with overall_score >0.7 AND all hard constraints met
- REFINE: Some good candidates exist but need improvement
- RESTART: <3 viable candidates OR fundamental strategy issues

REFINEMENT STRATEGIES:
1. Property-focused: Adjust specific properties (MW, logP, solubility)
2. Structure-focused: Modify scaffolds, functional groups
3. Constraint-focused: Relax non-critical constraints
4. Strategy-focused: Change generation approach (scaffold→fragment)

ACCEPTANCE THRESHOLDS:
- Minimum candidates: 5
- Minimum overall score: 0.7
- Maximum safety violations: 0
- Constraint compliance: 100% for hard constraints

OUTPUT ONLY JSON:
{{
  "decision": "accept|refine|restart",
  "reasoning": "Detailed justification for decision",
  "accepted_candidates": [
    {{
      "smiles": "SMILES_string",
      "rank": 1,
      "overall_score": 0.85,
      "rationale": "Why this candidate is promising"
    }}
  ],
  "refinement_instructions": {{
    "strategy": "property|structure|constraint|strategy",
    "specific_changes": [
      "Reduce molecular weight by 50-100 Da",
      "Improve solubility by adding polar groups",
      "Maintain core scaffold structure"
    ],
    "priority_properties": ["MW", "logS"],
    "relaxed_constraints": []
  }},
  "iteration_summary": {{
    "improvements_made": ["Changes since last iteration"],
    "remaining_issues": ["Outstanding problems"],
    "progress_score": 0.75
  }},
  "recommendations": [
    "Suggestions for future iterations or alternative approaches"
  ],
  "confidence": 0.88
}}

CRITICAL RULES:
- Be constructive: Always provide specific, actionable feedback
- Balance quality vs. quantity: Don't accept poor candidates just to meet numbers
- Consider practical synthesis: Academic interest vs. real-world applicability
- Track iteration progress: Are we improving or stuck in local optima?
- Fail gracefully: If max iterations reached, accept best available candidates

ITERATION LIMITS:
- Iteration 1: Be somewhat lenient, expect refinement
- Iteration 2: Moderate standards, look for improvement
- Iteration 3: Accept best available, provide honest assessment
""",
        agent=agent,
        expected_output='JSON with decision, reasoning, accepted_candidates, refinement_instructions, and recommendations'
    )