import json
import time
from crewai import Crew
from agents.parser_agent import create_parser_agent, create_parsing_task
from agents.generator_agent import create_generator_agent, create_generation_task
from agents.validator_agent import create_validator_agent, create_validation_task
from agents.critic_agent import create_critic_agent, create_critic_task
from utils.json_parsing_utils import safe_json_parse, format_generation_results


class PipelineRunner:
    def __init__(self, seed=None, delay_between_calls=2):
        self.seed = seed
        self.delay = delay_between_calls
        self.debug_mode = True

    def _add_delay(self):
        """Add delay between API calls to prevent rate limiting"""
        import random
        base_delay = self.delay
        actual_delay = base_delay + random.uniform(0, 1)

        if self.debug_mode:
            print(f"🕐 Waiting {actual_delay:.1f}s")

        time.sleep(actual_delay)

    def _debug_print(self, message):
        """Print debug messages if debug mode is enabled"""
        if self.debug_mode:
            print(f"🐛 DEBUG: {message}")

    def _safe_format_generation_results(self, generator_result):
        """Safely format generation results with error handling"""
        try:
            formatted = format_generation_results(generator_result)

            # Ensure it's a dictionary
            if isinstance(formatted, str):
                self._debug_print("⚠️ format_generation_results returned string, attempting to parse as JSON")
                try:
                    formatted = json.loads(formatted)
                except json.JSONDecodeError:
                    self._debug_print("❌ Failed to parse formatted results as JSON, creating fallback structure")
                    formatted = {
                        "smiles_list": [],
                        "detailed_results": [],
                        "design_strategy": "Error in generation formatting",
                        "target_analysis": "Error in generation formatting"
                    }

            # Ensure required keys exist
            if not isinstance(formatted, dict):
                formatted = {
                    "smiles_list": [],
                    "detailed_results": [],
                    "design_strategy": "Error in generation formatting",
                    "target_analysis": "Error in generation formatting"
                }

            # Add missing keys with defaults
            defaults = {
                "smiles_list": [],
                "detailed_results": [],
                "design_strategy": "",
                "target_analysis": "",
                "feedback_addressed": "",
                "historical_learning": "",
                "novelty_confirmation": "",
                "iteration_evolution": "",
                "pattern_recognition": ""
            }

            for key, default_value in defaults.items():
                if key not in formatted:
                    formatted[key] = default_value

            return formatted

        except Exception as e:
            self._debug_print(f"❌ Error in _safe_format_generation_results: {e}")
            return {
                "smiles_list": [],
                "detailed_results": [],
                "design_strategy": f"Error in generation: {str(e)}",
                "target_analysis": "Error in generation formatting",
                "feedback_addressed": "",
                "historical_learning": "",
                "novelty_confirmation": "",
                "iteration_evolution": "",
                "pattern_recognition": ""
            }

    def _extract_critic_feedback(self, critic_result):
        """Extract meaningful feedback from critic response"""
        if not isinstance(critic_result, dict):
            return None

        # Try multiple sources of feedback
        feedback_parts = []

        # 1. Generation feedback (primary)
        if critic_result.get("generation_feedback"):
            feedback_parts.append(f"GENERATION FEEDBACK: {critic_result['generation_feedback']}")

        # 2. Specific suggestions
        if critic_result.get("specific_suggestions"):
            suggestions = critic_result["specific_suggestions"]
            if isinstance(suggestions, list):
                feedback_parts.append(f"SPECIFIC SUGGESTIONS: {'; '.join(suggestions)}")

        # 3. Key improvements needed
        if critic_result.get("key_improvements_needed"):
            improvements = critic_result["key_improvements_needed"]
            if isinstance(improvements, list):
                feedback_parts.append(f"KEY IMPROVEMENTS: {'; '.join(improvements)}")

        # 4. Top recommendation
        if critic_result.get("top_recommendation"):
            feedback_parts.append(f"TOP RECOMMENDATION: {critic_result['top_recommendation']}")

        # 5. Summary feedback
        if critic_result.get("summary"):
            feedback_parts.append(f"SUMMARY: {critic_result['summary']}")

        # 6. Weaknesses from ranked molecules
        if critic_result.get("ranked"):
            weaknesses = []
            for mol in critic_result["ranked"]:
                if mol.get("weaknesses"):
                    weaknesses.extend(mol["weaknesses"])
            if weaknesses:
                feedback_parts.append(f"ADDRESS WEAKNESSES: {'; '.join(set(weaknesses))}")

        return "\n\n".join(feedback_parts) if feedback_parts else None

    def _extract_key_issues(self, critic_result):
        """Extract key issues from critic response for history tracking"""
        if not isinstance(critic_result, dict):
            return []

        issues = []

        # Extract common issues from ranked molecules
        if critic_result.get("ranked"):
            for mol in critic_result["ranked"]:
                if mol.get("weaknesses"):
                    issues.extend(mol["weaknesses"])

        # Extract from improvement suggestions
        if critic_result.get("key_improvements_needed"):
            improvements = critic_result["key_improvements_needed"]
            if isinstance(improvements, list):
                issues.extend(improvements)

        # Extract from specific suggestions
        if critic_result.get("specific_suggestions"):
            suggestions = critic_result["specific_suggestions"]
            if isinstance(suggestions, list):
                issues.extend(suggestions)

        return list(set(issues))  # Remove duplicates

    def _summarize_generation_for_history(self, formatted_gen_results):
        """Create a concise summary of generation results for history"""
        try:
            summary = {
                "design_strategy": formatted_gen_results.get("design_strategy", ""),
                "target_analysis": formatted_gen_results.get("target_analysis", ""),
                "total_candidates": len(formatted_gen_results.get("smiles_list", []))
            }

            # Extract key reasoning points
            detailed_results = formatted_gen_results.get("detailed_results", [])
            if detailed_results and isinstance(detailed_results, list):
                reasoning_points = []
                for result in detailed_results:
                    if isinstance(result, dict) and result.get("reasoning"):
                        # Extract first sentence of reasoning
                        first_sentence = result["reasoning"].split('.')[0]
                        reasoning_points.append(first_sentence[:100])
                summary["key_reasoning"] = reasoning_points[:3]  # Top 3
            else:
                summary["key_reasoning"] = []

            return summary
        except Exception as e:
            self._debug_print(f"Error in _summarize_generation_for_history: {e}")
            return {
                "design_strategy": "Error in summarization",
                "target_analysis": "Error in summarization",
                "total_candidates": 0,
                "key_reasoning": []
            }

    def run_single_shot(self, user_input: str):
        """Run single-shot pipeline: parse -> generate -> validate"""
        try:
            print("🔍 Starting single-shot pipeline...")

            # Step 1: Parse
            print("Step 1: Parsing query...")
            parser = create_parser_agent(llm_seed=self.seed)
            parse_task = create_parsing_task(user_input, parser)
            parser_crew = Crew(agents=[parser], tasks=[parse_task], verbose=False)
            parser_result = parser_crew.kickoff().raw
            parsed_spec = safe_json_parse(parser_result, "Parser")
            self._add_delay()

            # Step 2: Generate
            print("Step 2: Generating candidates...")
            generator = create_generator_agent(llm_seed=self.seed)
            gen_task = create_generation_task(json.dumps(parsed_spec), generator)
            generator_crew = Crew(agents=[generator], tasks=[gen_task], verbose=False)
            generator_result = self._run_agent_with_retry(generator_crew)
            formatted_gen_results = self._safe_format_generation_results(generator_result)
            self._add_delay()

            # Track all generated molecules (before validation)
            all_generated_smiles = formatted_gen_results.get("smiles_list", [])
            if not all_generated_smiles:
                # Try to extract from candidates
                candidates = formatted_gen_results.get("candidates", [])
                all_generated_smiles = [c.get("smiles") for c in candidates if c.get("smiles")]

            if not all_generated_smiles:
                return {
                    "error": "No SMILES generated",
                    "valid": [],
                    "invalid": [],
                    "total_generated": 0,
                    "generation_details": formatted_gen_results
                }

            print(f"Generated {len(all_generated_smiles)} total molecules")

            # Step 3: Validate
            print("Step 3: Validating molecules...")
            validator = create_validator_agent(llm_seed=self.seed)
            val_task = create_validation_task(
                all_generated_smiles,
                json.dumps(parsed_spec),
                validator
            )
            validator_crew = Crew(agents=[validator], tasks=[val_task], verbose=False)
            validator_result = validator_crew.kickoff().raw
            validated = safe_json_parse(validator_result, "Validator")
            self._add_delay()

            valid_smiles = validated.get("valid", [])
            invalid_smiles = validated.get("invalid", [])

            # Ensure all molecules are accounted for
            accounted_molecules = set(valid_smiles + invalid_smiles)
            missing_molecules = [mol for mol in all_generated_smiles if mol not in accounted_molecules]

            # Add missing molecules as invalid (likely parsing issues)
            if missing_molecules:
                invalid_smiles.extend(missing_molecules)
                print(f"Added {len(missing_molecules)} unaccounted molecules as invalid")

            total_generated = len(all_generated_smiles)
            total_valid = len(valid_smiles)
            total_invalid = len(invalid_smiles)

            print(f"✅ Single-shot complete: {total_generated} generated, {total_valid} valid, {total_invalid} invalid")

            return {
                "valid": valid_smiles,
                "invalid": invalid_smiles,
                "total_generated": total_generated,
                "generation_details": formatted_gen_results,
                "parsed_spec": parsed_spec,
                "validation_details": validated.get("validation_details", {})
            }

        except Exception as e:
            print(f"❌ Single-shot pipeline failed: {e}")
            return {"error": str(e), "valid": [], "invalid": [], "total_generated": 0}

    def run_iterative(self, user_input: str, max_iterations=3):
        """Run iterative pipeline with proper molecule tracking"""
        try:
            print("🔄 Starting iterative pipeline...")

            # Step 1: Parse (once)
            print("Step 1: Parsing query...")
            parser = create_parser_agent(llm_seed=self.seed)
            parse_task = create_parsing_task(user_input, parser)
            parser_crew = Crew(agents=[parser], tasks=[parse_task], verbose=False)
            parser_result = parser_crew.kickoff().raw
            parsed_spec = safe_json_parse(parser_result, "Parser")
            self._add_delay()

            best_results = None
            all_iterations = []
            critic_feedback = None
            generation_history = []

            for iteration in range(max_iterations):
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

                # Step 2: Generate
                print("Step 2: Generating candidates...")
                generator = create_generator_agent(llm_seed=self.seed)
                gen_task = create_generation_task(
                    json.dumps(parsed_spec),
                    generator,
                    critic_feedback=critic_feedback,
                    generation_history=generation_history,
                    iteration_number=iteration + 1
                )
                generator_crew = Crew(agents=[generator], tasks=[gen_task], verbose=False)
                generator_result = self._run_agent_with_retry(generator_crew)
                formatted_gen_results = self._safe_format_generation_results(generator_result)
                self._add_delay()

                # Track all generated molecules
                all_generated_smiles = formatted_gen_results.get("smiles_list", [])
                if not all_generated_smiles:
                    candidates = formatted_gen_results.get("candidates", [])
                    all_generated_smiles = [c.get("smiles") for c in candidates if c.get("smiles")]

                if not all_generated_smiles:
                    print("⚠️ No valid SMILES generated in this iteration")
                    continue

                print(f"Generated {len(all_generated_smiles)} molecules in iteration {iteration + 1}")

                # Step 3: Validate
                print("Step 3: Validating molecules...")
                validator = create_validator_agent(llm_seed=self.seed)
                val_task = create_validation_task(
                    all_generated_smiles,
                    json.dumps(parsed_spec),
                    validator
                )
                validator_crew = Crew(agents=[validator], tasks=[val_task], verbose=False)
                validator_result = validator_crew.kickoff().raw
                validated = safe_json_parse(validator_result, "Validator")
                self._add_delay()

                valid_smiles = validated.get("valid", [])
                invalid_smiles = validated.get("invalid", [])

                # Ensure molecule accounting
                accounted_molecules = set(valid_smiles + invalid_smiles)
                missing_molecules = [mol for mol in all_generated_smiles if mol not in accounted_molecules]
                if missing_molecules:
                    invalid_smiles.extend(missing_molecules)

                # Step 4: Critic
                print("Step 4: Getting critic feedback...")
                critic = create_critic_agent(llm_seed=self.seed)

                detailed_results = formatted_gen_results.get("detailed_results", [])
                if not detailed_results:
                    detailed_results = [{"smiles": smiles, "reasoning": "Generated molecule"}
                                        for smiles in all_generated_smiles]

                critic_task = create_critic_task(
                    json.dumps(validated),
                    json.dumps(detailed_results),
                    json.dumps(parsed_spec),
                    critic
                )
                critic_crew = Crew(agents=[critic], tasks=[critic_task], verbose=False)
                critic_result = critic_crew.kickoff().raw
                final_ranked = safe_json_parse(critic_result, "Critic")
                self._add_delay()

                # Extract critic feedback for next iteration
                new_feedback = self._extract_critic_feedback(final_ranked)
                critic_feedback = new_feedback

                # Build generation history entry
                history_entry = {
                    "iteration": iteration + 1,
                    "generated_smiles": all_generated_smiles,
                    "valid_molecules": valid_smiles,
                    "invalid_molecules": invalid_smiles,
                    "top_ranked": final_ranked.get("ranked", [])[:3] if final_ranked.get("ranked") else [],
                    "critic_feedback": critic_feedback,
                    "key_issues": self._extract_key_issues(final_ranked),
                    "generation_summary": self._summarize_generation_for_history(formatted_gen_results),
                    "best_score": final_ranked.get("ranked", [{}])[0].get("score", 0) if final_ranked.get(
                        "ranked") else 0
                }
                generation_history.append(history_entry)

                iteration_result = {
                    "iteration": iteration + 1,
                    "valid": valid_smiles,
                    "invalid": invalid_smiles,
                    "total_generated": len(all_generated_smiles),
                    "ranked": final_ranked.get("ranked", []),
                    "generation_details": formatted_gen_results,
                    "critic_feedback": critic_feedback,
                    "validation_details": validated.get("validation_details", {})
                }

                all_iterations.append(iteration_result)
                print(f"✅ Iteration {iteration + 1}: {len(all_generated_smiles)} generated, "
                      f"{len(valid_smiles)} valid, {len(invalid_smiles)} invalid")

                # Track best results
                if (best_results is None or
                        (iteration_result["ranked"] and
                         iteration_result["ranked"][0].get("score", 0) >
                         best_results["ranked"][0].get("score", 0) if best_results.get("ranked") else True)):
                    best_results = iteration_result

                # Early stopping
                if (iteration_result["ranked"] and
                        iteration_result["ranked"][0].get("score", 0) > 0.8 and
                        len(valid_smiles) >= 2):
                    print("🎯 Early stopping: Found excellent candidates")
                    break

            print(f"🏁 Iterative pipeline complete: {len(all_iterations)} iterations")

            # Aggregate all molecules across iterations
            all_valid = []
            all_invalid = []
            total_generated_count = 0

            for iter_result in all_iterations:
                all_valid.extend(iter_result["valid"])
                all_invalid.extend(iter_result["invalid"])
                total_generated_count += iter_result["total_generated"]

            return {
                "valid": list(set(all_valid)),  # Remove duplicates
                "invalid": list(set(all_invalid)),  # Remove duplicates
                "total_generated": total_generated_count,
                "ranked": best_results["ranked"] if best_results else [],
                "all_iterations": all_iterations,
                "parsed_spec": parsed_spec,
                "total_iterations": len(all_iterations),
                "best_iteration": best_results["iteration"] if best_results else None,
                "generation_history": generation_history
            }

        except Exception as e:
            print(f"❌ Iterative pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "valid": [], "invalid": [], "total_generated": 0}

    def _run_agent_with_retry(self, crew, max_retries=3):
        """Run agent with retry logic"""
        for attempt in range(max_retries):
            try:
                result = crew.kickoff()
                if result and result.raw and len(result.raw.strip()) > 10:
                    return result.raw
                print(f"⚠️ Attempt {attempt + 1}: Empty response, retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
        raise RuntimeError("All retry attempts failed")