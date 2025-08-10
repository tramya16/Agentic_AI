#!/usr/bin/env python3
"""
Automated Molecular Generation Experiment Suite
Simple orchestrator that runs experiments and evaluations
"""

import argparse
from pathlib import Path
from scripts.experiment_runner import AutomatedExperimentRunner
from scripts.oracle_evaluator import OracleEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run molecular generation experiments")
    parser.add_argument("--config", default="config/experiment_config.yaml", 
                       help="Path to experiment configuration file")
    parser.add_argument("--run-experiments", action="store_true", 
                       help="Run experiments")
    parser.add_argument("--evaluate-only", type=str, 
                       help="Only evaluate existing results (provide results directory)")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to run (e.g., gemini_2_0_flash deepseek_v3)")
    
    args = parser.parse_args()
    
    if args.evaluate_only:
        print("🔍 Running evaluation only...")
        evaluator = OracleEvaluator()
        
        results_dir = Path(args.evaluate_only)
        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            return
        
        # Evaluate results
        all_evaluations = evaluator.evaluate_results_directory(results_dir)
        
        if all_evaluations:
            # Create comparison visualizations
            viz_dir = results_dir / "comparative_analysis"
            comparison_df = evaluator.create_comparison_visualizations(all_evaluations, viz_dir)
            
            print(f"\n✅ Evaluation complete!")
            print(f"📊 Evaluated {len(all_evaluations)} configurations")
            print(f"📈 Visualizations saved to: {viz_dir}")
        else:
            print("❌ No evaluations could be performed")
        
        return
    
    if args.run_experiments:
        print("🚀 Starting automated experiment suite...")
        
        # Initialize experiment runner
        runner = AutomatedExperimentRunner(config_path=args.config)
        
        # Filter models if specified
        if args.models:
            original_models = runner.config['models'].copy()
            filtered_models = {k: v for k, v in original_models.items() if k in args.models}
            
            if not filtered_models:
                print(f"❌ No matching models found. Available: {list(original_models.keys())}")
                return
            
            runner.config['models'] = filtered_models
            print(f"🎯 Running experiments for: {list(filtered_models.keys())}")
        
        # Run experiments
        all_results, master_file = runner.run_all_experiments()
        
        if not all_results:
            print("❌ No experiments completed successfully")
            return
        
        print(f"\n✅ Experiments complete! Results saved to: {master_file}")
        
        # Run evaluation if enabled
        if runner.config['evaluation']['enabled']:
            print("\n🔍 Starting automatic evaluation...")
            
            evaluator = OracleEvaluator()
            
            # Get all results directories
            results_base = Path(runner.config['output']['base_results_dir'])
            
            # Evaluate each results directory
            all_evaluations = {}
            
            for model_name, model_results in all_results.items():
                for temp_config, temp_data in model_results.items():
                    results_dir = Path(temp_data['results_dir'])
                    
                    print(f"📊 Evaluating: {model_name} - {temp_config}")
                    
                    eval_results = evaluator.evaluate_results_directory(results_dir.parent)
                    if eval_results:
                        all_evaluations.update(eval_results)
            
            if all_evaluations:
                # Create comparative analysis
                comparative_dir = results_base / f"{runner.timestamp}_comparative_analysis"
                comparison_df = evaluator.create_comparison_visualizations(all_evaluations, comparative_dir)
                
                print(f"\n🎉 Complete pipeline finished!")
                print(f"📊 Experiments: {len(all_results)} models")
                print(f"📈 Evaluations: {len(all_evaluations)} configurations")
                print(f"📋 Comparative analysis: {comparative_dir}")
            else:
                print("⚠️ No evaluations could be performed")
    
    else:
        print("❌ Please specify --run-experiments or --evaluate-only")
        parser.print_help()


if __name__ == "__main__":
    main()
