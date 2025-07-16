# pipeline_runner.py

import json
from crewai import Crew
from agents.parser_agent import create_parser_agent, create_parsing_task
from agents.generator_agent import create_generator_agent, create_generation_task
from agents.validator_agent import create_validator_agent, create_validation_task
from agents.critic_agent import create_critic_agent, create_critic_task
from crew.crew_setup import safe_json_parse, format_generation_results, canonical


class PipelineRunner:
    def __init__(self, seed=None):
        self.seed = seed

    def run_single_shot(self, user_input: str):
        """Run single-shot pipeline: parse -> generate -> validate"""
        try:
            # Step 1: Parse
            parser = create_parser_agent(llm_seed=self.seed)
            parse_task = create_parsing_task(user_input, parser)
            parser_crew = Crew(agents=[parser], tasks=[parse_task], verbose=False)
            parser_result = parser_crew.kickoff().raw
            parsed_spec = safe_json_parse(parser_result, "Parser")

            # Step 2: Generate
            generator = create_generator_agent(llm_seed=self.seed)
            gen_task = create_generation_task(json.dumps(parsed_spec), generator)
            generator_crew = Crew(agents=[generator], tasks=[gen_task], verbose=False)
            generator_result = generator_crew.kickoff().raw
            formatted_gen_results = format_generation_results(generator_result)

            if not formatted_gen_results["smiles_list"]:
                return {"error": "No valid SMILES generated", "valid": [], "invalid": []}

            # Step 3: Validate
            validator = create_validator_agent(llm_seed=self.seed)
            val_task = create_validation_task(
                formatted_gen_results["smiles_list"],
                json.dumps(parsed_spec),
                validator
            )
            validator_crew = Crew(agents=[validator], tasks=[val_task], verbose=False)
            validator_result = validator_crew.kickoff().raw
            validated = safe_json_parse(validator_result, "Validator")

            return {
                "valid": validated.get("valid", []),
                "invalid": validated.get("invalid", []),
                "generation_details": formatted_gen_results,
                "parsed_spec": parsed_spec
            }

        except Exception as e:
            return {"error": str(e), "valid": [], "invalid": []}

    def run_iterative(self, user_input: str, max_iterations=3):
        """Run iterative pipeline with critic feedback"""
        try:
            # Step 1: Parse (once)
            parser = create_parser_agent(llm_seed=self.seed)
            parse_task = create_parsing_task(user_input, parser)
            parser_crew = Crew(agents=[parser], tasks=[parse_task], verbose=False)
            parser_result = parser_crew.kickoff().raw
            parsed_spec = safe_json_parse(parser_result, "Parser")

            best_results = None
            all_iterations = []

            for iteration in range(max_iterations):
                # Step 2: Generate
                generator = create_generator_agent(llm_seed=self.seed)
                gen_task = create_generation_task(json.dumps(parsed_spec), generator)
                generator_crew = Crew(agents=[generator], tasks=[gen_task], verbose=False)
                generator_result = generator_crew.kickoff().raw
                formatted_gen_results = format_generation_results(generator_result)

                if not formatted_gen_results["smiles_list"]:
                    continue

                # Step 3: Validate
                validator = create_validator_agent(llm_seed=self.seed)
                val_task = create_validation_task(
                    formatted_gen_results["smiles_list"],
                    json.dumps(parsed_spec),
                    validator
                )
                validator_crew = Crew(agents=[validator], tasks=[val_task], verbose=False)
                validator_result = validator_crew.kickoff().raw
                validated = safe_json_parse(validator_result, "Validator")

                # Step 4: Critic - Create molecule objects for critic
                valid_molecules = [{"smiles": smiles} for smiles in validated.get("valid", [])]

                critic = create_critic_agent(llm_seed=self.seed)
                critic_task = create_critic_task(
                    json.dumps({"valid": valid_molecules}),
                    json.dumps(formatted_gen_results["detailed_results"]),
                    json.dumps(parsed_spec),
                    critic
                )
                critic_crew = Crew(agents=[critic], tasks=[critic_task], verbose=False)
                critic_result = critic_crew.kickoff().raw
                final_ranked = safe_json_parse(critic_result, "Critic")

                iteration_result = {
                    "iteration": iteration + 1,
                    "valid": validated.get("valid", []),
                    "invalid": validated.get("invalid", []),
                    "ranked": final_ranked.get("ranked", []),
                    "generation_details": formatted_gen_results
                }

                all_iterations.append(iteration_result)

                # Keep track of best results (most valid molecules)
                if best_results is None or len(iteration_result["valid"]) > len(best_results["valid"]):
                    best_results = iteration_result

                # Early stopping if we have good results
                if len(validated.get("valid", [])) >= 3:
                    break

            return {
                "valid": best_results["valid"] if best_results else [],
                "invalid": best_results["invalid"] if best_results else [],
                "ranked": best_results["ranked"] if best_results else [],
                "all_iterations": all_iterations,
                "parsed_spec": parsed_spec,
                "total_iterations": len(all_iterations)
            }

        except Exception as e:
            return {"error": str(e), "valid": [], "invalid": [], "ranked": []}