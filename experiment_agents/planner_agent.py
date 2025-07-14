# agents/planner_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm

def create_planner_agent():
    llm = load_llm()
    return Agent(
        role='Molecular Strategy Planner',
        goal=(
            'Given a parsed molecular design specification, '
            'decide which downstream agent to invoke next: '
            '– “generator” (with a chosen strategy), '
            '– “optimizer”, or '
            '– fallback “general_chem”.'
        ),
        backstory=(
            'You are an expert computational chemist who plans the workflow '
            'by selecting the most appropriate specialist agent based on the design goals.'
        ),
        tools=[],  # Planner itself uses only the LLM
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

def create_planning_task(parsed_spec_json: str, agent):
    return Task(
        description=f"""
You are the Molecular Strategy Planner.  Here is the parsed specification:
{parsed_spec_json}

Decide which agent should handle this:
  • If the task is to propose new scaffolds or novel structures, choose “generator” and pick one strategy from [“scaffold”, “fragment”, “de-novo”].
  • If the task is to tweak existing candidates to meet property goals, choose “optimizer”.
  • Otherwise, choose “general_chem” for ad-hoc tool use.

Output ONLY a JSON object with these fields:
  1. next_agent: one of "generator", "optimizer", "general_chem"
  2. strategy: if next_agent=="generator", one of "scaffold","fragment","de-novo"
  3. reasoning: your brief justification

Use double quotes, no extra keys, no prose outside the JSON.
""",
        agent=agent,
        expected_output='JSON with keys next_agent, (strategy), reasoning'
    )
