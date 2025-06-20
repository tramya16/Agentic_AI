
from crewai import Crew
from agents.parser_agent import create_parser_agent,create_parsing_task

def run_molecular_parser(user_input: str):
    # Initialize agents
    parser_agent = create_parser_agent()

    # Create tasks
    parsing_task = create_parsing_task(user_input, parser_agent)

    # Assemble crew
    crew = Crew(
        agents=[parser_agent],
        tasks=[parsing_task],
        verbose=0
    )

    # Execute workflow
    return crew.kickoff()