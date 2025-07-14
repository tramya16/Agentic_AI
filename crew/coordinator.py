from __future__ import annotations

"""Optimized Coordinator for streamlined molecular design pipeline.

Key improvements:
- Conditional agent execution based on user needs
- Simplified context passing 
- Reduced prompt complexity
- Better error handling
- Compact outputs
"""

import json
from typing import Any, Dict, List, Optional

from crewai import Crew, Task

from agents.parser_agent import create_parser_agent, create_parsing_task
from agents.structure_agent import create_structure_agent, create_structure_task
from agents.property_agent import create_property_agent, create_property_task
from agents.fragment_agent import create_fragment_agent, create_fragment_task
from agents.generator_agent import create_final_generator_agent, create_final_generation_task
from agents.validator_agent import create_validator_agent, create_validation_task
from agents.critic_agent import create_critic_agent, create_critic_task


def _safe_json_load(text: Optional[str]):
    """Extract JSON from LLM response."""
    if not text:
        return None

    txt = text.strip()

    # Remove markdown fences
    if txt.startswith("```"):
        first_newline = txt.find("\n")
        if first_newline != -1:
            txt = txt[first_newline + 1:]
        if txt.endswith("```"):
            txt = txt[:-3]
        txt = txt.strip()

    # Extract JSON content
    if "{" in txt and "}" in txt:
        start = txt.find("{")
        end = txt.rfind("}") + 1
        txt = txt[start:end]

    try:
        return json.loads(txt)
    except Exception:
        return {"error": "Failed to parse JSON", "raw_text": text}


def _raw(output) -> Optional[str]:
    return getattr(output, "raw", None)


def _determine_needed_agents(parsed_spec: dict) -> List[str]:
    """Determine which agents are needed based on user request."""
    needed = []

    # Check if structural modifications are needed
    if (parsed_spec.get("optimization_goals") and
            any("similarity" in str(goal) for goal in parsed_spec.get("optimization_goals", []))):
        needed.append("structure")

    # Check if property optimization is needed
    if (parsed_spec.get("optimization_goals") and
            any("log" in str(goal).lower() or "tpsa" in str(goal).lower() or "qed" in str(goal).lower()
                for goal in parsed_spec.get("optimization_goals", []))):
        needed.append("property")

    # Check if fragment modifications are needed
    if (parsed_spec.get("constraints") and
            any("fragment" in str(constraint).lower() or "group" in str(constraint).lower()
                for constraint in parsed_spec.get("constraints", []))):
        needed.append("fragment")

    return needed


def run_molecular_pipeline(
        user_input: str,
        *,
        max_iterations: int = 2,  # Reduced from 3
        auto_detect_agents: bool = True,
        manual_agents: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run streamlined molecular design pipeline."""

    # Parse user request
    parser_agent = create_parser_agent()
    parse_task = create_parsing_task(user_input, parser_agent)
    parsed_output = Crew(agents=[parser_agent], tasks=[parse_task]).kickoff()
    parsed_spec = _safe_json_load(_raw(parsed_output))

    if not parsed_spec or "error" in parsed_spec:
        return {"status": "parse_error", "error": parsed_spec}

    # Determine needed agents
    if auto_detect_agents:
        needed_agents = _determine_needed_agents(parsed_spec)
    else:
        needed_agents = manual_agents or ["structure", "property"]

    print(f"Using agents: {needed_agents}")

    run_log = {
        "parsed_spec": parsed_spec,
        "agents_used": needed_agents,
        "iterations": []
    }

    # Create agents only if needed
    agents = {}
    if "structure" in needed_agents:
        agents["structure"] = create_structure_agent()
    if "property" in needed_agents:
        agents["property"] = create_property_agent()
    if "fragment" in needed_agents:
        agents["fragment"] = create_fragment_agent()

    agents["generator"] = create_final_generator_agent()
    agents["validator"] = create_validator_agent()
    agents["critic"] = create_critic_agent()

    # Iterative refinement
    refinement_context = None

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== ITERATION {iteration} ===")

        # Prepare iteration spec
        iteration_spec = parsed_spec.copy()
        if refinement_context:
            iteration_spec["refinement"] = refinement_context

        # Collect outputs from needed agents
        agent_outputs = {}

        # Structure analysis
        if "structure" in needed_agents:
            try:
                task = create_structure_task(iteration_spec, agents["structure"])
                output = Crew(agents=[agents["structure"]], tasks=[task]).kickoff()
                agent_outputs["structure"] = _safe_json_load(_raw(output))
            except Exception as e:
                agent_outputs["structure"] = {"error": str(e)}

        # Property optimization
        if "property" in needed_agents:
            try:
                task = create_property_task(
                    iteration_spec,
                    agent_outputs.get("structure", {}),
                    agents["property"]
                )
                output = Crew(agents=[agents["property"]], tasks=[task]).kickoff()
                agent_outputs["property"] = _safe_json_load(_raw(output))
            except Exception as e:
                agent_outputs["property"] = {"error": str(e)}

        # Fragment analysis
        if "fragment" in needed_agents:
            try:
                task = create_fragment_task(
                    iteration_spec,
                    agent_outputs.get("structure", {}),
                    agent_outputs.get("property", {}),
                    agents["fragment"]
                )
                output = Crew(agents=[agents["fragment"]], tasks=[task]).kickoff()
                agent_outputs["fragment"] = _safe_json_load(_raw(output))
            except Exception as e:
                agent_outputs["fragment"] = {"error": str(e)}

        # Generate molecules
        try:
            gen_task = create_final_generation_task(
                iteration_spec,
                agent_outputs.get("structure", {}),
                agent_outputs.get("property", {}),
                agent_outputs.get("fragment", {}),
                agents["generator"]
            )
            gen_output = Crew(agents=[agents["generator"]], tasks=[gen_task]).kickoff()
            gen_result = _safe_json_load(_raw(gen_output))
        except Exception as e:
            gen_result = {"error": str(e), "final_smiles": []}

        # Extract candidates
        candidates = []
        if isinstance(gen_result, dict) and "final_smiles" in gen_result:
            smiles = gen_result["final_smiles"]
            if isinstance(smiles, str):
                candidates = [smiles]
            elif isinstance(smiles, list):
                candidates = smiles

        # Validate candidates
        try:
            val_task = create_validation_task(candidates, iteration_spec, agents["validator"])
            val_output = Crew(agents=[agents["validator"]], tasks=[val_task]).kickoff()
            val_result = _safe_json_load(_raw(val_output))
        except Exception as e:
            val_result = {"error": str(e), "validated_candidates": []}

        # Critic evaluation
        try:
            critic_task = create_critic_task(val_result, iteration_spec, iteration, agents["critic"])
            critic_output = Crew(agents=[agents["critic"]], tasks=[critic_task]).kickoff()
            critic_result = _safe_json_load(_raw(critic_output))
        except Exception as e:
            critic_result = {"decision": "accept", "error": str(e)}

        # Log iteration
        iter_log = {
            "iteration": iteration,
            "agent_outputs": {k: v for k, v in agent_outputs.items() if v},
            "generated": gen_result,
            "validated": val_result,
            "critic": critic_result
        }
        run_log["iterations"].append(iter_log)

        # Decision logic
        decision = critic_result.get("decision", "accept")
        if decision == "accept":
            run_log["status"] = "accepted"
            run_log["final_result"] = critic_result
            break
        elif decision == "refine":
            refinement_context = critic_result.get("refinement_instructions", {})
            continue
        else:
            refinement_context = None
            continue
    else:
        run_log["status"] = "max_iterations"
        run_log["final_result"] = run_log["iterations"][-1]["critic"]

    return run_log


if __name__ == "__main__":
    query = input("Molecular design request: ").strip() or "Design a molecule similar to aspirin with better solubility"
    result = run_molecular_pipeline(query)

    if result["status"] == "accepted":
        print("\n=== SUCCESS ===")
        final = result["final_result"]
        if "accepted_candidates" in final:
            for i, candidate in enumerate(final["accepted_candidates"][:3]):
                print(f"{i + 1}. {candidate.get('smiles', 'N/A')} (score: {candidate.get('overall_score', 'N/A')})")
    else:
        print(f"\n=== RESULT: {result['status']} ===")
        print(f"Used agents: {result.get('agents_used', [])}")
        print(f"Iterations: {len(result.get('iterations', []))}")