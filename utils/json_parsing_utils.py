import json
import re
from typing import Dict, Any


def safe_json_parse(raw_response: str, agent_name: str) -> Dict[Any, Any]:
    """
    Safely parse JSON from agent responses, handling common formatting issues
    """
    if not raw_response:
        print(f"❌ {agent_name} returned empty response")
        return _get_fallback_structure(agent_name)

    # Clean the response first
    cleaned_response = raw_response.strip()

    try:
        # First, try direct parsing
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        print(f"⚠️ JSON parsing error in {agent_name}:")
        print(f"Raw response: {cleaned_response[:200]}...")

        # Method 1: Extract from markdown code blocks
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # ```json { } ```
            r'```\s*(\{.*?\})\s*```',  # ``` { } ```
            r'(\{.*?\})',  # Just find any { }
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned_response, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match.strip())
                    print(f"✅ Successfully extracted JSON using pattern")
                    return parsed
                except json.JSONDecodeError:
                    continue

        # Method 2: Try to fix common JSON issues
        fixed_attempts = [
            _fix_trailing_commas(cleaned_response),
            _fix_unquoted_keys(cleaned_response),
            _extract_largest_json_object(cleaned_response)
        ]

        for attempt in fixed_attempts:
            if attempt:
                try:
                    return json.loads(attempt)
                except json.JSONDecodeError:
                    continue

        print(f"❌ All JSON parsing attempts failed for {agent_name}")
        return _get_fallback_structure(agent_name)


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas that break JSON"""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _fix_unquoted_keys(text: str) -> str:
    """Try to fix unquoted JSON keys"""
    # This is a simple attempt - may not work for all cases
    text = re.sub(r'(\w+):', r'"\1":', text)
    return text


def _extract_largest_json_object(text: str) -> str:
    """Extract the largest valid JSON object from text"""
    best_json = ""
    max_length = 0

    # Find all potential JSON objects
    for i, char in enumerate(text):
        if char == '{':
            brace_count = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[i:j + 1]
                        if len(candidate) > max_length:
                            try:
                                json.loads(candidate)
                                best_json = candidate
                                max_length = len(candidate)
                            except:
                                pass
                        break
    return best_json


def _get_fallback_structure(agent_name: str) -> Dict[str, Any]:
    """Return appropriate fallback structure for each agent type"""
    fallbacks = {
        "Parser": {
            "target_molecules": [],
            "properties": {},
            "constraints": [],
            "task_type": "generation",
            "similarity_threshold": 0.6
        },
        "Generator": {
            "candidates": [],
            "design_strategy": "Error in generation - using fallback",
            "target_analysis": "Unable to analyze due to generation error",
            "feedback_addressed": "Error prevented feedback processing",
            "historical_learning": "Error prevented learning analysis",
            "novelty_confirmation": "Error prevented novelty check",
            "iteration_evolution": "Error prevented evolution tracking",
            "pattern_recognition": "Error prevented pattern analysis"
        },
        "Validator": {
            "valid": [],
            "invalid": [],
            "validation_details": {
                "duplicates_found": [],
                "target_matches": [],
                "invalid_smiles": [],
                "toxicity_flags": []
            }
        },
        "Critic": {
            "ranked": [],
            "summary": "Error in critic analysis",
            "top_recommendation": "Unable to provide recommendation due to error",
            "generation_feedback": "Error prevented feedback generation",
            "key_improvements_needed": [],
            "specific_suggestions": []
        }
    }
    return fallbacks.get(agent_name, {})


def format_generation_results(raw_result: str) -> Dict[str, Any]:
    """Format generation results into a standardized structure"""
    try:
        parsed = safe_json_parse(raw_result, "Generator")

        # Extract SMILES from candidates with better error handling
        smiles_list = []
        detailed_results = []

        if "candidates" in parsed and isinstance(parsed["candidates"], list):
            for candidate in parsed["candidates"]:
                if isinstance(candidate, dict) and "smiles" in candidate:
                    smiles_list.append(candidate["smiles"])
                    detailed_results.append(candidate)

        # Ensure all required keys exist
        required_keys = [
            "design_strategy", "target_analysis", "feedback_addressed",
            "historical_learning", "novelty_confirmation", "iteration_evolution",
            "pattern_recognition"
        ]

        for key in required_keys:
            if key not in parsed:
                parsed[key] = f"Missing {key} in generation output"

        return {
            "smiles_list": smiles_list,
            "detailed_results": detailed_results,
            "design_strategy": parsed.get("design_strategy", ""),
            "target_analysis": parsed.get("target_analysis", ""),
            "feedback_addressed": parsed.get("feedback_addressed", ""),
            "historical_learning": parsed.get("historical_learning", ""),
            "novelty_confirmation": parsed.get("novelty_confirmation", ""),
            "iteration_evolution": parsed.get("iteration_evolution", ""),
            "pattern_recognition": parsed.get("pattern_recognition", ""),
            "raw_response": raw_result
        }

    except Exception as e:
        print(f"❌ Error formatting generation results: {e}")
        return {
            "smiles_list": [],
            "detailed_results": [],
            "design_strategy": f"Error in generation: {str(e)}",
            "target_analysis": "Error in generation formatting",
            "feedback_addressed": "Error prevented feedback processing",
            "historical_learning": "Error prevented learning analysis",
            "novelty_confirmation": "Error prevented novelty check",
            "iteration_evolution": "Error prevented evolution tracking",
            "pattern_recognition": "Error prevented pattern analysis",
            "raw_response": raw_result
        }