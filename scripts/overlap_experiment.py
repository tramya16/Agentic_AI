# overlap_experiment.py

import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from pipeline_runner import PipelineRunner
from queries import get_query_list, get_query_prompt
from crew.crew_setup import canonical


class OverlapExperiment:
    def __init__(self, results_dir="experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def run_single_query_experiment(self, query_name, query_prompt, runs=5, top_n=5):
        """Run overlap experiment for a single query"""
        single_shot_results = []
        iterative_results = []

        for run in range(runs):
            seed = random.randint(0, 2 ** 32 - 1)
            runner = PipelineRunner(seed=seed)

            # Single-shot pipeline
            single_result = runner.run_single_shot(query_prompt)
            single_shot_results.append({
                "run": run + 1,
                "seed": seed,
                "result": single_result,
                "error": single_result.get("error")
            })

            # Iterative pipeline
            iterative_result = runner.run_iterative(query_prompt, max_iterations=3)
            iterative_results.append({
                "run": run + 1,
                "seed": seed,
                "result": iterative_result,
                "error": iterative_result.get("error")
            })

            print(f"  Run {run + 1}/{runs} completed")

        return {
            "query_name": query_name,
            "query_prompt": query_prompt,
            "single_shot": single_shot_results,
            "iterative": iterative_results,
            "metadata": {
                "runs": runs,
                "top_n": top_n,
                "timestamp": datetime.now().isoformat()
            }
        }

    def analyze_overlap_results(self, results, top_n=5):
        """Analyze overlap statistics for pipeline results"""

        def analyze_pipeline_results(pipeline_results, pipeline_name):
            all_smiles = []
            all_valid = []
            all_invalid = []
            successful_runs = 0

            for run_data in pipeline_results:
                result = run_data["result"]
                if result.get("error"):
                    continue

                successful_runs += 1

                # Extract molecules based on pipeline type
                if pipeline_name == "iterative" and "ranked" in result:
                    # For iterative, use ranked results if available
                    molecules = result["ranked"][:top_n]
                    smiles_list = [mol["smiles"] for mol in molecules if isinstance(mol, dict) and "smiles" in mol]
                    # Fall back to valid list if ranked doesn't work
                    if not smiles_list:
                        smiles_list = result.get("valid", [])[:top_n]
                else:
                    # For single-shot, use valid molecules
                    smiles_list = result.get("valid", [])[:top_n]

                # Track valid/invalid (now just lists of SMILES)
                all_valid.extend(result.get("valid", []))
                all_invalid.extend(result.get("invalid", []))

                # Canonicalize and collect
                canonical_smiles = [canonical(s) for s in smiles_list]
                all_smiles.extend([s for s in canonical_smiles if s])

            total_smiles = len(all_smiles)
            unique_smiles = len(set(all_smiles))
            total_valid = len(all_valid)
            total_invalid = len(all_invalid)
            total_molecules = total_valid + total_invalid

            return {
                "successful_runs": successful_runs,
                "total_smiles": total_smiles,
                "unique_smiles": unique_smiles,
                "overlap_percentage": 100 * (total_smiles - unique_smiles) / total_smiles if total_smiles > 0 else 0,
                "total_valid": total_valid,
                "total_invalid": total_invalid,
                "valid_percentage": 100 * total_valid / total_molecules if total_molecules > 0 else 0,
                "invalid_percentage": 100 * total_invalid / total_molecules if total_molecules > 0 else 0,
                "unique_molecules": list(set(all_smiles))
            }

        single_shot_stats = analyze_pipeline_results(results["single_shot"], "single_shot")
        iterative_stats = analyze_pipeline_results(results["iterative"], "iterative")

        return {
            "query_name": results["query_name"],
            "single_shot": single_shot_stats,
            "iterative": iterative_stats,
            "metadata": results["metadata"]
        }

    def run_full_experiment(self, query_names=None, runs=5, top_n=5):
        """Run overlap experiment for all specified queries"""
        if query_names is None:
            query_names = get_query_list()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []
        all_analyses = []

        print(f"Starting overlap experiment with {len(query_names)} queries, {runs} runs each")
        print(f"Timestamp: {timestamp}")

        for i, query_name in enumerate(query_names):
            print(f"\nProcessing query {i + 1}/{len(query_names)}: {query_name}")
            query_prompt = get_query_prompt(query_name)

            # Run experiment for this query
            query_results = self.run_single_query_experiment(
                query_name, query_prompt, runs=runs, top_n=top_n
            )

            # Analyze results
            analysis = self.analyze_overlap_results(query_results, top_n=top_n)

            all_results.append(query_results)
            all_analyses.append(analysis)

            # Save individual query results
            query_file = self.results_dir / f"{query_name}_{timestamp}.json"
            with open(query_file, 'w') as f:
                json.dump(query_results, f, indent=2)

            print(
                f"  Single-shot: {analysis['single_shot']['unique_smiles']}/{analysis['single_shot']['total_smiles']} unique ({analysis['single_shot']['overlap_percentage']:.1f}% overlap)")
            print(
                f"  Iterative: {analysis['iterative']['unique_smiles']}/{analysis['iterative']['total_smiles']} unique ({analysis['iterative']['overlap_percentage']:.1f}% overlap)")

        # Save summary results
        summary_file = self.results_dir / f"experiment_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "analyses": all_analyses,
                "metadata": {
                    "total_queries": len(query_names),
                    "runs_per_query": runs,
                    "top_n": top_n,
                    "timestamp": timestamp
                }
            }, f, indent=2)

        # Generate summary table
        self.generate_summary_table(all_analyses, timestamp)

        return all_results, all_analyses

    def generate_summary_table(self, analyses, timestamp):
        """Generate a readable summary table"""
        summary_lines = []
        summary_lines.append(f"Overlap Experiment Summary - {timestamp}")
        summary_lines.append("=" * 80)
        summary_lines.append("")

        # Headers
        summary_lines.append(
            f"{'Query':<20} {'Pipeline':<12} {'Runs':<5} {'Valid%':<8} {'Unique':<8} {'Total':<8} {'Overlap%':<8}")
        summary_lines.append("-" * 80)

        # Statistics collectors
        single_shot_overlaps = []
        iterative_overlaps = []
        single_shot_valid_pcts = []
        iterative_valid_pcts = []

        for analysis in analyses:
            query_name = analysis["query_name"][:19]  # Truncate for display

            # Single-shot row
            ss = analysis["single_shot"]
            summary_lines.append(
                f"{query_name:<20} {'Single-shot':<12} {ss['successful_runs']:<5} {ss['valid_percentage']:<7.1f} {ss['unique_smiles']:<8} {ss['total_smiles']:<8} {ss['overlap_percentage']:<7.1f}")

            # Iterative row
            it = analysis["iterative"]
            summary_lines.append(
                f"{'':<20} {'Iterative':<12} {it['successful_runs']:<5} {it['valid_percentage']:<7.1f} {it['unique_smiles']:<8} {it['total_smiles']:<8} {it['overlap_percentage']:<7.1f}")
            summary_lines.append("")

            # Collect stats
            if ss['total_smiles'] > 0:
                single_shot_overlaps.append(ss['overlap_percentage'])
                single_shot_valid_pcts.append(ss['valid_percentage'])
            if it['total_smiles'] > 0:
                iterative_overlaps.append(it['overlap_percentage'])
                iterative_valid_pcts.append(it['valid_percentage'])

        # Overall statistics
        summary_lines.append("=" * 80)
        summary_lines.append("OVERALL STATISTICS")
        summary_lines.append("=" * 80)

        if single_shot_overlaps:
            summary_lines.append(
                f"Single-shot Overlap: Mean={np.mean(single_shot_overlaps):.1f}%, Std={np.std(single_shot_overlaps):.1f}%")
            summary_lines.append(
                f"Single-shot Valid%:  Mean={np.mean(single_shot_valid_pcts):.1f}%, Std={np.std(single_shot_valid_pcts):.1f}%")

        if iterative_overlaps:
            summary_lines.append(
                f"Iterative Overlap:   Mean={np.mean(iterative_overlaps):.1f}%, Std={np.std(iterative_overlaps):.1f}%")
            summary_lines.append(
                f"Iterative Valid%:    Mean={np.mean(iterative_valid_pcts):.1f}%, Std={np.std(iterative_valid_pcts):.1f}%")

        # Save summary table
        summary_file = self.results_dir / f"summary_table_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))

        # Print to console
        print("\n" + '\n'.join(summary_lines))


if __name__ == "__main__":
    # Run experiment
    experiment = OverlapExperiment()

    # Test with first 3 queries for demo
    test_queries = ["albuterol_similarity", "amlodipine_mpo", "celecoxib_rediscovery"]

    try:
        results, analyses = experiment.run_full_experiment(
            query_names=test_queries,
            runs=3,  # Use fewer runs for testing
            top_n=5
        )
        print("\nExperiment completed successfully!")

    except Exception as e:
        print(f"Experiment failed: {e}")