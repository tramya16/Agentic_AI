from crewai import Crew, Task
from agents.parser_agent import create_parser_agent, create_parsing_task
from agents.structure_agent import create_structure_agent, create_structure_task
from agents.property_agent import create_property_agent, create_property_task
from agents.fragment_agent import create_fragment_agent, create_fragment_task
from agents.generator_agent import create_final_generator_agent, create_final_generation_task  # NEW IMPORT
import json

def run_molecular_pipeline(user_input: str, steps=["parser", "structure", "property", "fragment", "generator"]):
    print("=== Initializing Agents ===")
    parser_agent = create_parser_agent()
    structure_agent = create_structure_agent()
    property_agent = create_property_agent()
    fragment_agent = create_fragment_agent()
    final_agent = create_final_generator_agent()  # NEW AGENT

    # --- Parser Stage ---
    print("=== Running Parser ===")
    parse_task = create_parsing_task(user_input, parser_agent)
    parse_crew = Crew(agents=[parser_agent], tasks=[parse_task], manager_llm=None)
    parsed_result_crew_output = parse_crew.kickoff()

    parsed_data_for_next_tasks = {}
    try:
        parsed_data_for_next_tasks = json.loads(parsed_result_crew_output.raw)
    except json.JSONDecodeError:
        print("Warning: Parser output was not valid JSON. Subsequent agents might not get correct input.")

    print(f"[DEBUG] Parsed result type: {type(parsed_result_crew_output)}")

    # --- Structure Stage ---
    structure_result = {}  # Initialize outside for scope
    if "structure" in steps:
        print("=== Running Structure Agent ===")
        structure_task = create_structure_task(
            parsed_json=parsed_data_for_next_tasks,
            agent=structure_agent,
            context_tasks=[parse_task]  # Pass the parse_task object for context
        )
        structure_crew = Crew(agents=[structure_agent], tasks=[structure_task], manager_llm=None)
        structure_crew_output = structure_crew.kickoff()
        try:
            # Assuming structure_crew_output.raw contains JSON string
            structure_result = json.loads(structure_crew_output.raw)
        except json.JSONDecodeError:
            print("Warning: Structure agent output was not valid JSON.")
            structure_result = {"raw_output": structure_crew_output.raw}  # Fallback
    else:
        # If structure step is skipped, still create a dummy structure_task
        structure_task = Task(description="Skipped structure analysis.", agent=structure_agent, expected_output="Skipped.", context=[parse_task])

    # --- Property Stage ---
    property_result = {}  # Initialize outside for scope
    if "property" in steps:
        print("=== Running Property Agent ===")
        property_task = create_property_task(
            parsed_json=parsed_data_for_next_tasks,
            structure_output=structure_result,  # Pass the extracted dict
            agent=property_agent,
            context_tasks=[parse_task, structure_task]  # Pass both preceding Task objects for context
        )
        property_crew = Crew(agents=[property_agent], tasks=[property_task], manager_llm=None)
        property_crew_output = property_crew.kickoff()
        try:
            property_result = json.loads(property_crew_output.raw)
        except json.JSONDecodeError:
            print("Warning: Property agent output was not valid JSON.")
            property_result = {"raw_output": property_crew_output.raw}  # Fallback
    else:
        # Create a dummy task if step is skipped
        property_task = Task(description="Skipped property optimization.", agent=property_agent, expected_output="Skipped.", context=[parse_task, structure_task])

    # --- Fragment Stage ---
    fragment_result = {}  # Initialize outside for scope
    if "fragment" in steps:
        print("=== Running Fragment Agent ===")
        fragment_task = create_fragment_task(
            parsed_json=parsed_data_for_next_tasks,
            structure_output=structure_result,
            property_output=property_result,  # Pass the extracted dict
            agent=fragment_agent,
            context_tasks=[parse_task, structure_task, property_task]  # Pass all preceding Task objects
        )
        fragment_crew = Crew(agents=[fragment_agent], tasks=[fragment_task], manager_llm=None)
        fragment_crew_output = fragment_crew.kickoff()
        try:
            fragment_result = json.loads(fragment_crew_output.raw)
        except json.JSONDecodeError:
            print("Warning: Fragment agent output was not valid JSON.")
            fragment_result = {"raw_output": fragment_crew_output.raw}  # Fallback
    else:
        # Create a dummy task if step is skipped
        fragment_task = Task(description="Skipped fragment analysis.", agent=fragment_agent, expected_output="Skipped.", context=[parse_task, structure_task, property_task])

    # --- NEW: Final Generator Stage ---
    final_result = {}  # Initialize outside for scope
    if "generator" in steps:
        print("=== Running Final Molecule Generator ===")
        final_task = create_final_generation_task(
            parsed_json=parsed_data_for_next_tasks,
            structure_output=structure_result,
            property_output=property_result,
            fragment_output=fragment_result,
            agent=final_agent,
            context_tasks=[parse_task, structure_task, property_task, fragment_task]  # All previous tasks
        )
        final_crew = Crew(agents=[final_agent], tasks=[final_task], manager_llm=None)
        final_crew_output = final_crew.kickoff()
        try:
            final_result = json.loads(final_crew_output.raw)
        except json.JSONDecodeError:
            print("Warning: Final generator output was not valid JSON.")
            final_result = {"raw_output": final_crew_output.raw}  # Fallback
    else:
        # Create a dummy task if step is skipped
        final_task = Task(description="Skipped final molecule generation.", agent=final_agent, expected_output="Skipped.", context=[parse_task, structure_task, property_task, fragment_task])

    return {
        "parsed": parsed_result_crew_output,
        "structure": structure_crew_output if "structure" in steps else structure_task,
        "property": property_crew_output if "property" in steps else property_task,
        "fragment": fragment_crew_output if "fragment" in steps else fragment_task,
        "final": final_crew_output if "final" in steps else final_task  # NEW FINAL OUTPUT
    }