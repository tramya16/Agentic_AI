# agents/tool_runner_agent.py

from crewai import Agent, Task
from llms.model_loader import load_llm
from tools.tool_registry import ALL_TOOLS_FLAT

def create_tool_runner_agent():
    llm = load_llm()
    return Agent(
        role='Tool Runner',
        goal=(
            'Given explicit tool invocation instructions, execute the specified tool '
            'with exact arguments and return its output.'
        ),
        backstory=(
            'You are a reliable execution engine that never alters user-provided commands.'
        ),
        tools=ALL_TOOLS_FLAT,
        verbose=True,
        llm=llm,
        allow_delegation=False
    )

def create_tool_task(tool_name: str, args_json: str, agent):
    return Task(
        description=f"""
You are the Tool Runner.  
Invoke the tool named "{tool_name}" with these arguments:
{args_json}

Return ONLY the tool’s JSON/string output, nothing else.
""",
        agent=agent,
        expected_output='Raw tool output'
    )
