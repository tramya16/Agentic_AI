# agents/generator_agent.py - IMPROVED VERSION

from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import GENERATOR_TOOLS
import json


def create_generator_agent(llm_seed: int):
    llm = load_llm(seed=llm_seed)
    return Agent(
        role="Molecule Generator",
        goal="Generate novel molecular structures based on specifications and feedback",
        backstory="""You are a computational chemist expert at designing molecules. You learn from 
        feedback and avoid repeating previous molecules. You ALWAYS return valid JSON format.""",
        tools=GENERATOR_TOOLS,
        verbose=True,
        llm=llm,
        allow_delegation=False,
        max_execution_time=300,
        max_retry=3  # Increased retries
    )


def create_generation_task(parsed_spec: str, agent: Agent, critic_feedback: str = None,
                           generation_history: list = None, iteration_number: int = 1):
    # Simplified feedback section
    feedback_section = ""
    if critic_feedback:
        feedback_section = f"""
PREVIOUS FEEDBACK TO ADDRESS:
{critic_feedback[:500]}...

You MUST incorporate this feedback into your new designs.
"""

    # Simplified history section
    history_section = ""
    if generation_history:
        previous_smiles = []
        key_issues = []

        for hist in generation_history:
            previous_smiles.extend(hist.get('generated_smiles', []))
            key_issues.extend(hist.get('key_issues', []))

        history_section = f"""
PREVIOUS MOLECULES TO AVOID (DO NOT REPEAT):
{', '.join(previous_smiles[:10])}

KEY ISSUES TO ADDRESS:
{', '.join(set(key_issues[:5]))}
"""

    return Task(
        description=f"""
Generate 3-5 novel molecules based on:

SPECIFICATION: {parsed_spec}
{feedback_section}
{history_section}

CRITICAL: Return ONLY valid JSON in this EXACT format:
{{
  "candidates": [
    {{
      "smiles": "CC(C)O",
      "reasoning": "Brief explanation of design rationale",
      "modifications": ["change1", "change2"],
      "expected_improvements": ["improvement1", "improvement2"],
      "learning_applied": "How previous feedback was used",
      "novelty_justification": "Why this is different from previous molecules"
    }}
  ],
  "design_strategy": "Overall approach for this iteration",
  "target_analysis": "Analysis of target molecules",
  "feedback_addressed": "How feedback was incorporated",
  "historical_learning": "What was learned from previous iterations",
  "novelty_confirmation": "Confirmation of novelty vs previous generations",
  "iteration_evolution": "How approach evolved",
  "pattern_recognition": "Patterns identified from history"
}}

REQUIREMENTS:
- Generate only VALID SMILES strings
- Make molecules DIFFERENT from previous iterations
- Address feedback specifically
- Use tools to verify novelty with duplicate_check_tool
- Keep JSON format EXACT - no extra text, no markdown
""",
        agent=agent,
        expected_output="Valid JSON object with candidate molecules and analysis"
    )