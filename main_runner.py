"""
Main runner for LLM molecular generation experiments
Supports running experiments, oracle scoring, and comparisons
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime

import config
# Import configuration
from config import ExperimentConfig, validate_config

# Import experiment classes (with minimal modifications)
from scripts.Experiment import ExperimentOne
from scripts.improved_queries import get_query_list

# Import scoring and comparison (will be imported when needed)
# from oracle_scorer import ComprehensiveOracleEvaluator
# from corrected_oracle_evaluation import ResearchFocusedLLMComparator


class ExperimentRunner:
    def __init__(self):
        self.config = ExperimentConfig

    def run_single_model_experiment(self, model_key):
        """Run experiment for a single model"""
        if model_key not in self.config.MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        model_config = self.config.MODELS[model_key]
        results_dir = self.config.get_model_result_dir(model_key)

        print(f"\n🚀 Running experiment for {model_config['display_name']}")
        print(f"📁 Results will be saved to: {results_dir}")

        # Create modified ExperimentOne with config
        experiment = ExperimentOne(
            results_dir=str(results_dir),
            model_name=model_config['model_id'],
        )

        try:
            query_list = config.ExperimentConfig.QUERY_LIST
            print(f"📋 Running {len(query_list)} queries with {self.config.RUNS_PER_QUERY} runs each")

            results, analyses = experiment.run_comprehensive_experiment(
                query_names=query_list,
                runs=self.config.RUNS_PER_QUERY
            )

            print(f"✅ {model_config['display_name']} experiment completed successfully!")
            return True

        except Exception as e:
            print(f"❌ {model_config['display_name']} experiment failed: {e}")
            traceback.print_exc()
            return False

    def run_all_models_experiment(self):
        """Run experiments for all configured models"""
        print(f"🚀 Starting experiments for all {len(self.config.MODELS)} models")

        success_count = 0
        failed_models = []

        for model_key in self.config.MODELS.keys():
            try:
                if self.run_single_model_experiment(model_key):
                    success_count += 1
                else:
                    failed_models.append(model_key)
            except Exception as e:
                print(f"❌ Critical error with {model_key}: {e}")
                failed_models.append(model_key)

        print(f"\n📊 Experiment Summary:")
        print(f"  ✅ Successful: {success_count}/{len(self.config.MODELS)}")
        if failed_models:
            print(f"  ❌ Failed: {failed_models}")

        return success_count == len(self.config.MODELS)

    def run_oracle_scoring_single_model(self, model_key):
        """Run oracle scoring for a single model"""
        if model_key not in self.config.MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        model_config = self.config.MODELS[model_key]
        model_dir = self.config.get_model_result_dir(model_key)

        print(f"\n🔮 Running oracle scoring for {model_config['display_name']}")
        print(f"📁 Scoring results in: {model_dir}")

        if not model_dir.exists():
            print(f"❌ Model directory doesn't exist: {model_dir}")
            return False

        # Check if directory has results
        json_files = list(model_dir.glob("*.json"))
        if not json_files:
            print(f"❌ No JSON files found in {model_dir}")
            return False

        print(f"📄 Found {len(json_files)} result files")

        try:
            # Import oracle scorer
            from oracle_scorer import ComprehensiveOracleEvaluator

            # Use the full path as expected by oracle scorer
            evaluator = ComprehensiveOracleEvaluator(results_dir=str(model_dir))
            results = evaluator.run_complete_evaluation()

            if results:
                print(f"✅ Oracle scoring completed for {model_config['display_name']}!")
                return True
            else:
                print(f"❌ Oracle scoring failed for {model_config['display_name']}!")
                return False

        except ImportError as e:
            print(f"❌ Could not import oracle scorer: {e}")
            return False
        except Exception as e:
            print(f"❌ Oracle scoring failed for {model_config['display_name']}: {e}")
            traceback.print_exc()
            return False

    def run_oracle_scoring(self):
        """Run oracle scoring on all models"""
        print(f"\n🔮 Starting oracle scoring for all models...")

        success_count = 0
        failed_models = []

        for model_key in self.config.MODELS.keys():
            try:
                if self.run_oracle_scoring_single_model(model_key):
                    success_count += 1
                else:
                    failed_models.append(model_key)
            except Exception as e:
                print(f"❌ Critical error scoring {model_key}: {e}")
                failed_models.append(model_key)

        print(f"\n📊 Oracle Scoring Summary:")
        print(f"  ✅ Successful: {success_count}/{len(self.config.MODELS)}")
        if failed_models:
            print(f"  ❌ Failed: {failed_models}")

        return success_count > 0  # Return True if at least one model was scored

    def run_llm_comparison(self):
        """Run LLM comparison analysis"""
        print(f"\n📊 Starting LLM comparison analysis...")
        print(f"📁 Analyzing results in: {self.config.RESULTS_DIR}")

        # Check if we have oracle-scored results
        oracle_scored_models = []
        for model_key in self.config.MODELS.keys():
            model_dir = self.config.get_model_result_dir(model_key)
            if model_dir.exists():
                # Check for oracle-scored files
                json_files = list(model_dir.glob("*.json"))
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        if self._has_oracle_scores(data):
                            oracle_scored_models.append(model_key)
                            break
                    except:
                        continue

        if not oracle_scored_models:
            print("❌ No oracle-scored results found. Run oracle scoring first.")
            return False

        print(f"📊 Found oracle scores for {len(oracle_scored_models)} models: {oracle_scored_models}")

        try:
            # Import LLM comparator
            from llm_comparisions import ResearchFocusedLLMComparator

            comparator = ResearchFocusedLLMComparator(base_dir=str(self.config.RESULTS_DIR))
            results = comparator.run_research_focused_analysis()

            if not results:
                print("❌ LLM comparison failed!")
                return False
            
            print("✅ LLM comparison completed successfully!")
            
            # Run additional visualizations
            print("\n📊 Running additional research question visualizations...")
            
            try:
                from ResearchQuestionsVisualizations import ResearchQuestionAnalyzer
                
                analyzer = ResearchQuestionAnalyzer(base_dir=str(self.config.RESULTS_DIR))
                research_results = analyzer.run_research_question_analysis()
                
                if research_results:
                    print("✅ Research question visualizations completed successfully!")
                else:
                    print("⚠️ Research question visualizations failed, but continuing...")
                    
            except Exception as e:
                print(f"⚠️ Research question visualizations failed: {e}, but continuing...")
            
            # Run MT-MOL comparison visualization
            print("\n📊 Running MT-MOL comparison visualization...")
            
            try:
                import subprocess
                result = subprocess.run(['python', 'compareAgainstMtMol.py'],
                                     capture_output=True, text=True, cwd='.')
                if result.returncode == 0:
                    print("✅ MT-MOL comparison visualization completed successfully!")
                    if result.stdout:
                        print(result.stdout)
                else:
                    print(f"⚠️ MT-MOL comparison visualization failed: {result.stderr}, but continuing...")
                    
            except Exception as e:
                print(f"⚠️ MT-MOL comparison visualization failed: {e}, but continuing...")
            
            return True

        except ImportError as e:
            print(f"❌ Could not import LLM comparator: {e}")
            return False
        except Exception as e:
            print(f"❌ LLM comparison failed: {e}")
            traceback.print_exc()
            return False

    def _has_oracle_scores(self, data):
        """Check if the JSON data contains oracle scores"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and 'single_shot' in value and 'iterative' in value:
                    # Check if it has the oracle scoring structure
                    for pipeline in ['single_shot', 'iterative']:
                        if 'runs' in value[pipeline]:
                            runs = value[pipeline]['runs']
                            if runs and isinstance(runs, list):
                                first_run = runs[0]
                                if 'molecules' in first_run and first_run['molecules']:
                                    first_mol = first_run['molecules'][0]
                                    if 'Oracle_Score' in first_mol:
                                        return True
        return False

    def run_scoring_and_comparison(self):
        """Run both oracle scoring and LLM comparison"""
        print(f"\n🔮📊 Starting scoring and comparison...")

        # Step 1: Run oracle scoring
        if not self.run_oracle_scoring():
            print("❌ Oracle scoring failed, stopping")
            return False

        # Step 2: Run LLM comparison
        if not self.run_llm_comparison():
            print("❌ LLM comparison failed, stopping")
            return False

        print("✅ Scoring and comparison completed successfully!")
        return True

    def run_full_pipeline(self):
        """Run the complete pipeline: experiments -> scoring -> comparison"""
        print("🚀 Starting full pipeline...")

        # Step 1: Run experiments
        if not self.run_all_models_experiment():
            print("❌ Experiments failed, stopping pipeline")
            return False

        # Step 2: Run scoring and comparison
        if not self.run_scoring_and_comparison():
            print("❌ Scoring/comparison failed, stopping pipeline")
            return False

        print("🎉 Full pipeline completed successfully!")
        return True


class ModifiedExperimentOne(ExperimentOne):
    """Modified ExperimentOne to use config parameters"""

    def __init__(self, results_dir, model_name, temperature, max_tokens):
        super().__init__(results_dir=results_dir)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run_single_query_experiment(self, query_name, query_data, runs=None, top_n=None):
        """Override to use config parameters"""
        if runs is None:
            runs = ExperimentConfig.RUNS_PER_QUERY
        if top_n is None:
            top_n = ExperimentConfig.TOP_N

        return super().run_single_query_experiment(query_name, query_data, runs, top_n)


def main():
    parser = argparse.ArgumentParser(description="LLM Molecular Generation Experiment Runner")
    parser.add_argument("action", choices=[
        "experiment", "single", "score", "score-single", "compare", "analysis", "full"
    ], help="Action to perform")

    parser.add_argument("--model", choices=list(ExperimentConfig.MODELS.keys()),
                       help="Specific model to run (for 'single' or 'score-single' actions)")

    parser.add_argument("--validate", action="store_true",
                       help="Validate configuration before running")

    args = parser.parse_args()

    # Validate configuration
    if args.validate or args.action not in ["score", "score-single", "compare", "analysis"]:
        if not validate_config():
            print("❌ Configuration validation failed")
            sys.exit(1)

    # Ensure directories
    ExperimentConfig.ensure_directories()

    # Create runner
    runner = ExperimentRunner()

    # Execute requested action
    try:
        if args.action == "experiment":
            success = runner.run_all_models_experiment()
        elif args.action == "single":
            if not args.model:
                print("❌ --model required for 'single' action")
                sys.exit(1)
            success = runner.run_single_model_experiment(args.model)
        elif args.action == "score":
            success = runner.run_oracle_scoring()
        elif args.action == "score-single":
            if not args.model:
                print("❌ --model required for 'score-single' action")
                sys.exit(1)
            success = runner.run_oracle_scoring_single_model(args.model)
        elif args.action == "compare":
            success = runner.run_llm_comparison()
        elif args.action == "analysis":
            success = runner.run_scoring_and_comparison()
        elif args.action == "full":
            success = runner.run_full_pipeline()
        else:
            print(f"❌ Unknown action: {args.action}")
            sys.exit(1)

        if success:
            print(f"\n✅ {args.action.title()} completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ {args.action.title()} failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()