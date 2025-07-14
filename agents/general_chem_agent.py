# agents/general_chem_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import ALL_TOOLS_FLAT

def create_general_chem_agent():
    llm = load_llm()
    return Agent(
        role='General Chemistry Assistant',
        goal=(
            'Handle ad-hoc requests and tooling for chemistry tasks that fall outside '
            'strict parsing, generation, or optimization.'
        ),
        backstory=(
            'You are a flexible chemical informatics assistant, able to run any tool on demand.'
        ),
        tools=ALL_TOOLS_FLAT,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

def create_general_chem_task(user_request: str, agent):
    return Task(
        description=f"""
You are the General Chemistry Assistant. The user asks:
{user_request}

Choose and run the appropriate tool(s) to fulfill the request.  
Return ONLY the raw tool outputs or a brief JSON summary if you combine multiple calls.
""",
        agent=agent,
        expected_output='Tool output or summary'
    )
