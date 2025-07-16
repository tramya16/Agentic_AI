# crew_setup.py

import json
import re
import random

from crewai import Crew

from agents.parser_agent import create_parser_agent, create_parsing_task
from agents.generator_agent import create_generator_agent, create_generation_task
from agents.validator_agent import create_validator_agent, create_validation_task
from agents.critic_agent import create_critic_agent, create_critic_task
from utils.chemistry_utils import check_duplicate

def extract_json_from_response(response):
    """Extract JSON from agent response, handling markdown code blocks"""
    # Look for JSON in code blocks
    json_pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block, try to find JSON directly
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        return match.group(0).strip()

    # Try array format for simple lists
    json_pattern = r'\[.*\]'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        return match.group(0).strip()

    return None


def safe_json_parse(response, step_name):
    """Safely parse JSON from agent response with error handling"""
    try:
        json_str = extract_json_from_response(response)
        if not json_str:
            raise ValueError(f"No JSON found in {step_name} response")

        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in {step_name}:")
        print(f"Raw response: {response}")
        print(f"Extracted JSON string: {json_str}")
        raise RuntimeError(f"{step_name} failed: {e}")
    except Exception as e:
        print(f"Error in {step_name}: {e}")
        print(f"Raw response: {response}")
        raise RuntimeError(f"{step_name} failed: {e}")


def format_generation_results(generation_result):
    """Format generation results for better readability and next agent consumption"""
    try:
        # Parse the rich generation output
        data = safe_json_parse(generation_result, "Generation formatting")

        # Handle both old format (simple array) and new format (rich object)
        if isinstance(data, list):
            # Old format - just SMILES strings
            return {
                "smiles_list": data,
                "detailed_results": {"candidates": [{"smiles": s, "reasoning": "Legacy format"} for s in data]},
                "formatted_summary": f"Generated {len(data)} candidates using legacy format"
            }

        # New format - rich object with reasoning
        if "candidates" in data:
            candidate_smiles = [candidate["smiles"] for candidate in data["candidates"]]

            # Create a formatted summary for the next agent
            formatted_summary = f"""
Generation Results:

Target Analysis: {data.get('target_analysis', 'Not provided')}

Design Strategy: {data.get('design_strategy', 'Not provided')}

Generated Candidates:
"""

            for i, candidate in enumerate(data["candidates"], 1):
                formatted_summary += f"""
{i}. SMILES: {candidate['smiles']}
   Reasoning: {candidate.get('reasoning', 'Not provided')}
   Modifications: {', '.join(candidate.get('modifications', []))}
   Expected Improvements: {', '.join(candidate.get('expected_improvements', []))}
"""

            return {
                "smiles_list": candidate_smiles,
                "detailed_results": data,
                "formatted_summary": formatted_summary
            }

        # Fallback if format is unexpected
        raise ValueError("Unexpected generation result format")

    except Exception as e:
        print(f"Error formatting generation results: {e}")
        print(f"Raw result: {generation_result}")
        return {
            "smiles_list": [],
            "detailed_results": {},
            "formatted_summary": "Error processing generation results"
        }


def run_pipeline(user_input: str, *, seed: int | None = None):
    """Run the complete molecular design pipeline"""
    results = {}

    # Step 1: Parser
    print("Step 1: Parsing query...")
    parser = create_parser_agent(llm_seed=seed)
    parse_task = create_parsing_task(user_input, parser)

    parser_crew = Crew(
        agents=[parser],
        tasks=[parse_task],
        verbose=False
    )
    parser_result = parser_crew.kickoff().raw
    results['parser_output'] = parser_result

    parsed_spec = safe_json_parse(parser_result, "Parser")

    # Step 2: Generator
    print("Step 2: Generating candidates...")
    generator = create_generator_agent(llm_seed=seed)
    gen_task = create_generation_task(json.dumps(parsed_spec), generator)

    generator_crew = Crew(
        agents=[generator],
        tasks=[gen_task],
        verbose=False
    )
    generator_result = generator_crew.kickoff().raw
    results['generator_output'] = generator_result

    # Format generation results for better handling
    formatted_gen_results = format_generation_results(generator_result)

    if not formatted_gen_results["smiles_list"]:
        raise RuntimeError("No valid SMILES generated")

    print(f"Generated {len(formatted_gen_results['smiles_list'])} candidates")
    print("Generation summary:")
    print(formatted_gen_results["formatted_summary"])

    # Step 3: Validator
    print("Step 3: Validating molecules...")
    validator = create_validator_agent(llm_seed=seed)
    val_task = create_validation_task(
        formatted_gen_results["smiles_list"],
        json.dumps(parsed_spec),
        validator
    )

    validator_crew = Crew(
        agents=[validator],
        tasks=[val_task],
        verbose=False
    )
    validator_result = validator_crew.kickoff().raw
    results['validator_output'] = validator_result

    validated = safe_json_parse(validator_result, "Validator")

    # Step 4: Critic
    print("Step 4: Ranking molecules...")
    critic = create_critic_agent(llm_seed=seed)

    # Create enhanced critic input with generation context

    critic_task = create_critic_task(
        json.dumps(validated),
        json.dumps(formatted_gen_results["detailed_results"]),
        json.dumps(parsed_spec),
        critic
    )

    critic_crew = Crew(
        agents=[critic],
        tasks=[critic_task],
        verbose=False
    )
    critic_result = critic_crew.kickoff().raw
    results['critic_output'] = critic_result

    final_ranked = safe_json_parse(critic_result, "Critic")

    results['final_ranked'] = final_ranked
    results['generation_details'] = formatted_gen_results
    return results


def canonical(smiles: str) -> str | None:
    """Return canonical SMILES or None if invalid."""
    out = check_duplicate(smiles, smiles)           # self-check ⇢ canonical form
    return out.get("canonical_smiles_1") if out.get("valid", True) else None

def overlap_for_single_query(prompt: str, runs: int = 5, top_n: int = 5):
    """Run the pipeline *runs* times for the same prompt and measure canonical overlap."""
    all_smiles = []

    for r in range(runs):
        seed = random.randint(0, 2**32 - 1)            # new seed per run
        result = run_pipeline(prompt, seed=seed)

        top = [
            canonical(mol["smiles"])                   # canonicalise
            for mol in result["final_ranked"]["ranked"][:top_n]
        ]
        all_smiles.extend([s for s in top if s])       # skip None (invalid)

    total  = len(all_smiles)
    unique = len(set(all_smiles))
    return {
        "runs": runs,
        "top_n_considered": top_n,
        "total_smiles": total,
        "unique_smiles": unique,
        "overlap_percentage": 100 * (total - unique) / total if total else 0,
        "all_smiles": all_smiles,
    }




if __name__ == "__main__":
    # Single run example
    # print("Single pipeline run:")
    # query = "Design a molecule similar to albuterol with better selectivity"
    # try:
    #     result = run_pipeline(query)
    #     print("\nFinal Results:")
    #     print(json.dumps(result['final_ranked'], indent=2))
    #
    #     print("\nGeneration Details:")
    #     print(result['generation_details']['formatted_summary'])
    #
    # except Exception as e:
    #     print(f"Pipeline failed: {e}")

    print("\n" + "=" * 50)
    print("Overlap Experiment:")

    # Run overlap experiment
    try:
        prompt = "Design a molecule similar to albuterol while preserving key functional groups."
        stats = overlap_for_single_query(prompt, runs=5, top_n=5)

        print(f"\n{stats['unique_smiles']}/{stats['total_smiles']} unique "
              f"({stats['overlap_percentage']:.1f}% overlap) in top-{stats['top_n_considered']} "
              f"across {stats['runs']} runs.")
        print("All canonical SMILES observed:", stats["all_smiles"])

    except Exception as e:
        print(f"Overlap experiment failed: {e}")