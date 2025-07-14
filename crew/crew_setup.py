# crew_setup.py

import json
from crewai import Crew

from agents.parser_agent import create_parser_agent, create_parsing_task
from experiment_agents.planner_agent import create_planner_agent, create_planning_task
from agents.generator_agent import create_generator_agent, create_generation_task
from agents.property_agent import create_optimizer_agent, create_optimization_task
from agents.general_chem_agent import create_general_chem_agent, create_general_chem_task


def run_full_pipeline(user_input: str):
    # Step 1: Run Parser
    parser = create_parser_agent()
    parse_task = create_parsing_task(user_input, parser)

    parser_crew = Crew(
        agents=[parser],
        tasks=[parse_task],
        verbose=0
    )
    parser_result = parser_crew.kickoff().raw

    # Extract parsed_spec as a dict
    try:
        parsed_output = json.loads(parser_result)
        parsed_spec = parsed_output['parsed_spec']
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Parser failed to return a valid parsed_spec JSON: {e}\nRaw output:\n{parser_result}")

    # Step 2: Run Planner (pass dict directly, not a JSON string)
    planner = create_planner_agent()
    plan_task = create_planning_task(parsed_spec, planner)

    planner_crew = Crew(
        agents=[planner],
        tasks=[plan_task],
        verbose=0
    )
    planner_result = planner_crew.kickoff().raw

    # Parse planner decision
    try:
        plan_output = json.loads(planner_result)
        next_agent = plan_output['next_agent']
        strategy = plan_output.get('strategy', '')
    except (json.JSONDecodeError, KeyError) as e:
        # Fallback to general_chem if planning fails
        next_agent = 'general_chem'
        strategy = ''

    # Step 3: Dispatch to the chosen agent
    if next_agent == 'generator':
        generator = create_generator_agent()
        # generator expects the spec serialized, so re-dump
        gen_task = create_generation_task(json.dumps(parsed_spec), strategy, generator)
        final_crew = Crew(agents=[generator], tasks=[gen_task], verbose=0)

    elif next_agent == 'optimizer':
        optimizer = create_optimizer_agent()
        # optimizer_agent.create_optimization_task expects Python lists/dicts
        candidates = parsed_spec.get("target_molecules", [])
        goals = parsed_spec.get("optimization_goals", [])
        opt_task = create_optimization_task(candidates, goals, optimizer)
        final_crew = Crew(agents=[optimizer], tasks=[opt_task], verbose=0)

    else:  # general_chem fallback
        general = create_general_chem_agent()
        general_task = create_general_chem_task(user_input, general)
        final_crew = Crew(agents=[general], tasks=[general_task], verbose=0)

    final_raw = final_crew.kickoff().raw

    return {
        'parser_result': parser_result,
        'planner_result': planner_result,
        'final_result': final_raw
    }


if __name__ == "__main__":
    user_input = "Design a molecule similar to albuterol while preserving key functional groups."
    print("Running full pipeline...\n")
    result = run_full_pipeline(user_input)
    print(json.dumps(result, indent=2))
