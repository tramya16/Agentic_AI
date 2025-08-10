import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from pipeline_runner import PipelineRunner
from improved_queries import get_query_data, get_query_list
from molecular_metrics import MolecularMetrics, MetricsVisualizer
import matplotlib.pyplot as plt
import os
import yaml


class AutomatedExperimentRunner:
    def __init__(self, config_path="config/experiment_config.yaml"):
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_calculator = MolecularMetrics()
        self.visualizer = MetricsVisualizer()
        
    def _load_config(self, config_path):
        """Load experiment configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_results_directory(self, model_name, model_id, temperature):
        """Create timestamped results directory"""
        model_clean = model_id.split('/')[-1].replace('-', '_').replace('.', '_')
        temp_str = f"temp_{str(temperature).replace('.', '_')}"
        
        dir_name = f"{self.timestamp}_{model_clean}_{temp_str}"
        results_dir = Path(self.config['output']['base_results_dir']) / dir_name
        
        # Create subdirectories
        (results_dir / "raw_results").mkdir(parents=True, exist_ok=True)
        (results_dir / "evaluations").mkdir(parents=True, exist_ok=True)
        (results_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        
        return results_dir
    
    def _get_query_list(self):
        """Get list of queries to run"""
        if self.config['queries']['selection'] == "all":
            return get_query_list()
        else:
            return self.config['queries']['selection']
    
    def run_single_query_experiment(self, query_name, query_data, results_dir, model_config):
        """Run experiment for a single query"""
        runs = self.config['experiment']['runs_per_query']
        max_iterations = self.config['experiment']['max_iterations']
        
        single_shot_results = []
        iterative_results = []

        query_prompt = query_data.get("prompt", "")
        target_smiles = query_data.get("target_smiles", "")
        reference_smiles = [target_smiles] if target_smiles else []

        print(f"🔬 Running experiment for: {query_name}")
        print(f"🎯 Target SMILES: {target_smiles}")

        for run in range(runs):
            seed = random.randint(0, 2 ** 32 - 1)
            runner = PipelineRunner(seed=seed, model_config=model_config)

            print(f"  📊 Run {run + 1}/{runs} (seed: {seed})")

            # Single-shot pipeline
            print("    🔄 Running single-shot...")
            single_result = runner.run_single_shot(query_prompt)

            single_molecules_data = self._extract_molecule_data(single_result)
            single_metrics = self.metrics_calculator.comprehensive_evaluation(
                single_molecules_data["all_generated"], reference_smiles
            )

            single_shot_results.append({
                "run": run + 1,
                "seed": seed,
                "result": single_result,
                "molecules_data": single_molecules_data,
                "metrics": single_metrics,
                "error": single_result.get("error")
            })

            # Iterative pipeline
            print("    🔄 Running iterative...")
            iterative_result = runner.run_iterative(query_prompt, max_iterations=max_iterations)

            iterative_molecules_data = self._extract_molecule_data(iterative_result)
            iterative_metrics = self.metrics_calculator.comprehensive_evaluation(
                iterative_molecules_data["all_generated"], reference_smiles
            )

            iterative_results.append({
                "run": run + 1,
                "seed": seed,
                "result": iterative_result,
                "molecules_data": iterative_molecules_data,
                "metrics": iterative_metrics,
                "error": iterative_result.get("error")
            })

            print(f"    ✅ Single-shot: {single_molecules_data['total_generated']} generated, "
                  f"{single_molecules_data['total_valid']} valid")
            print(f"    ✅ Iterative: {iterative_molecules_data['total_generated']} generated, "
                  f"{iterative_molecules_data['total_valid']} valid")

        return {
            "query_name": query_name,
            "query_data": query_data,
            "single_shot": single_shot_results,
            "iterative": iterative_results,
            "metadata": {
                "runs": runs,
                "max_iterations": max_iterations,
                "timestamp": self.timestamp,
                "model_config": model_config
            }
        }

    def _extract_molecule_data(self, pipeline_result):
        """Extract comprehensive molecule data from pipeline results"""
        if pipeline_result.get("error"):
            return {
                "all_generated": [],
                "valid_molecules": [],
                "invalid_molecules": [],
                "total_generated": 0,
                "total_valid": 0,
                "total_invalid": 0,
                "best_molecules": []
            }

        all_generated = []
        valid_molecules = []
        invalid_molecules = []
        best_molecules = []

        # Handle single-shot results
        if "valid" in pipeline_result and "all_iterations" not in pipeline_result:
            valid_molecules = pipeline_result.get("valid", [])
            invalid_molecules = pipeline_result.get("invalid", [])
            all_generated = valid_molecules + invalid_molecules
            best_molecules = valid_molecules[:10]

        # Handle iterative results
        elif "all_iterations" in pipeline_result:
            all_iteration_valid = []
            all_iteration_invalid = []

            for iteration in pipeline_result["all_iterations"]:
                iter_valid = iteration.get("valid", [])
                iter_invalid = iteration.get("invalid", [])
                all_iteration_valid.extend(iter_valid)
                all_iteration_invalid.extend(iter_invalid)

            all_generated = all_iteration_valid + all_iteration_invalid
            valid_molecules = all_iteration_valid
            invalid_molecules = all_iteration_invalid

            # Get best molecules from ranked results
            if pipeline_result.get("ranked"):
                best_molecules = []
                for mol in pipeline_result["ranked"][:10]:
                    if isinstance(mol, dict) and mol.get("smiles"):
                        best_molecules.append(mol["smiles"])
                    elif isinstance(mol, str):
                        best_molecules.append(mol)
            else:
                best_molecules = valid_molecules[:10]

        # Remove duplicates while preserving order
        def preserve_order_unique(lst):
            seen = set()
            result = []
            for item in lst:
                if item and item not in seen:
                    seen.add(item)
                    result.append(item)
            return result

        return {
            "all_generated": preserve_order_unique(all_generated),
            "valid_molecules": preserve_order_unique(valid_molecules),
            "invalid_molecules": preserve_order_unique(invalid_molecules),
            "total_generated": len(preserve_order_unique(all_generated)),
            "total_valid": len(preserve_order_unique(valid_molecules)),
            "total_invalid": len(preserve_order_unique(invalid_molecules)),
            "best_molecules": preserve_order_unique(best_molecules)
        }

    def run_model_experiment(self, model_name, model_config):
        """Run experiments for a specific model configuration"""
        print(f"\n{'='*80}")
        print(f"🚀 Starting experiments for {model_name}")
        print(f"📋 Model ID: {model_config['model_id']}")
        print(f"🌡️ Temperatures: {model_config['temperatures']}")
        print(f"{'='*80}")
        
        model_results = {}
        
        for temperature in model_config['temperatures']:
            print(f"\n🌡️ Running with temperature: {temperature}")
            
            # Create results directory for this configuration
            results_dir = self._create_results_directory(model_name, model_config['model_id'], temperature)
            print(f"📁 Results directory: {results_dir}")
            
            # Prepare model config for pipeline
            pipeline_model_config = {
                "name": model_config['name'],
                "model_id": model_config['model_id'],
                "temperature": temperature
            }
            
            # Get queries to run
            query_names = self._get_query_list()
            print(f"📝 Running {len(query_names)} queries")
            
            temp_results = []
            
            for i, query_name in enumerate(query_names):
                print(f"\n📊 Query {i+1}/{len(query_names)}: {query_name}")
                
                try:
                    query_data = get_query_data(query_name)
                    
                    # Run experiment
                    experiment_result = self.run_single_query_experiment(
                        query_name, query_data, results_dir, pipeline_model_config
                    )
                    
                    temp_results.append(experiment_result)
                    
                    # Save individual result if configured
                    if self.config['output']['save_individual_runs']:
                        result_file = results_dir / "raw_results" / f"{query_name}_detailed.json"
                        with open(result_file, 'w') as f:
                            json.dump(experiment_result, f, indent=2, default=str)
                    
                except Exception as e:
                    print(f"❌ Failed to process query {query_name}: {e}")
                    continue
            
            # Save temperature results
            temp_key = f"temp_{str(temperature).replace('.', '_')}"
            model_results[temp_key] = {
                "temperature": temperature,
                "results": temp_results,
                "results_dir": str(results_dir),
                "metadata": {
                    "model_name": model_name,
                    "model_id": model_config['model_id'],
                    "timestamp": self.timestamp,
                    "total_queries": len(temp_results)
                }
            }
            
            # Save comprehensive results
            summary_file = results_dir / "experiment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(model_results[temp_key], f, indent=2, default=str)
            
            print(f"✅ Completed {len(temp_results)} queries for {model_name} @ temp {temperature}")
        
        return model_results

    def run_all_experiments(self):
        """Run experiments for all configured models"""
        print(f"🚀 Starting automated experiment suite")
        print(f"⏰ Timestamp: {self.timestamp}")
        print(f"📊 Experiment: {self.config['experiment']['name']}")
        
        all_results = {}
        
        for model_name, model_config in self.config['models'].items():
            try:
                model_results = self.run_model_experiment(model_name, model_config)
                all_results[model_name] = model_results
                
            except Exception as e:
                print(f"❌ Failed to run experiments for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save master results file
        master_results_file = Path(self.config['output']['base_results_dir']) / f"{self.timestamp}_master_results.json"
        with open(master_results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n🎉 All experiments completed!")
        print(f"📊 Processed {len(all_results)} models")
        print(f"💾 Master results: {master_results_file}")
        
        return all_results, master_results_file
