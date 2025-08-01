import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from pipeline_runner import PipelineRunner
from improved_queries import get_query_data,get_query_list
from molecular_metrics import MolecularMetrics, MetricsVisualizer
import matplotlib.pyplot as plt
import os


class ExperimentOne:
    def __init__(self, results_dir="Gemini2.0_Flash_Temp_0.9_Results"):
        self.results_dir = Path(results_dir)
        # Ensure directory exists with proper permissions
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Verify directory is writable
        if not os.access(self.results_dir, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {self.results_dir}")

        self.metrics_calculator = MolecularMetrics()
        self.visualizer = MetricsVisualizer()

        print(f"📁 Results directory: {self.results_dir.absolute()}")

    def run_single_query_experiment(self, query_name, query_data, runs=3, top_n=5):
        """Run improved experiment for a single query with comprehensive metrics"""
        single_shot_results = []
        iterative_results = []

        query_prompt = query_data.get("prompt", "")
        target_smiles = query_data.get("target_smiles", "")
        reference_smiles = [target_smiles] if target_smiles else []

        print(f"Running experiment for: {query_name}")
        print(f"Target SMILES: {target_smiles}")

        for run in range(runs):
            seed = random.randint(0, 2 ** 32 - 1)
            runner = PipelineRunner(seed=seed)

            print(f"  Run {run + 1}/{runs} (seed: {seed})")

            # Single-shot pipeline
            print("    Running single-shot...")
            single_result = runner.run_single_shot(query_prompt)

            # Extract and track all molecules properly
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
            print("    Running iterative...")
            iterative_result = runner.run_iterative(query_prompt, max_iterations=3)

            # Extract and track all molecules properly
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

            print(f"    Single-shot: {single_molecules_data['total_generated']} generated, "
                  f"{single_molecules_data['total_valid']} valid, {single_molecules_data['total_invalid']} invalid")
            print(f"    Iterative: {iterative_molecules_data['total_generated']} generated, "
                  f"{iterative_molecules_data['total_valid']} valid, {iterative_molecules_data['total_invalid']} invalid")

        return {
            "query_name": query_name,
            "query_data": query_data,
            "single_shot": single_shot_results,
            "iterative": iterative_results,
            "metadata": {
                "runs": runs,
                "top_n": top_n,
                "timestamp": datetime.now().isoformat()
            }
        }

    # FIXED: _extract_molecule_data method in ExperimentOne class
    def _extract_molecule_data(self, pipeline_result):
        """Extract comprehensive molecule data from pipeline results - FIXED VERSION"""
        if pipeline_result.get("error"):
            return {
                "all_generated": [],
                "valid_molecules": [],
                "invalid_molecules": [],
                "total_generated": 0,
                "total_valid": 0,
                "total_invalid": 0,
                "best_molecules": [],
                "generation_breakdown": {}
            }

        all_generated = []
        valid_molecules = []
        invalid_molecules = []
        best_molecules = []
        generation_breakdown = {}

        # Handle single-shot results
        if "valid" in pipeline_result and "invalid" in pipeline_result and "all_iterations" not in pipeline_result:
            valid_molecules = pipeline_result.get("valid", [])
            invalid_molecules = pipeline_result.get("invalid", [])

            # FIXED: Properly reconstruct all_generated from valid + invalid
            all_generated = valid_molecules + invalid_molecules

            # Get best molecules (valid molecules, ordered by appearance)
            best_molecules = valid_molecules[:10]  # Top 10 for single-shot

            generation_breakdown = {
                "single_generation": {
                    "generated": len(all_generated),
                    "valid": len(valid_molecules),
                    "invalid": len(invalid_molecules)
                }
            }

        # Handle iterative results
        elif "all_iterations" in pipeline_result:
            iteration_data = {}
            all_iteration_valid = []
            all_iteration_invalid = []
            all_iteration_generated = []

            for iteration in pipeline_result["all_iterations"]:
                iter_num = iteration["iteration"]
                iter_valid = iteration.get("valid", [])
                iter_invalid = iteration.get("invalid", [])

                # FIXED: Track all molecules from this iteration
                all_iteration_valid.extend(iter_valid)
                all_iteration_invalid.extend(iter_invalid)
                all_iteration_generated.extend(iter_valid + iter_invalid)

                iteration_data[f"iteration_{iter_num}"] = {
                    "generated": len(iter_valid) + len(iter_invalid),
                    "valid": len(iter_valid),
                    "invalid": len(iter_invalid)
                }

            # FIXED: Use aggregated data
            all_generated = all_iteration_generated
            valid_molecules = all_iteration_valid
            invalid_molecules = all_iteration_invalid
            generation_breakdown = iteration_data

            # FIXED: Get best molecules from ranked results, fallback to valid molecules
            if pipeline_result.get("ranked"):
                best_molecules = []
                for mol in pipeline_result["ranked"][:10]:  # Top 10
                    if isinstance(mol, dict) and mol.get("smiles"):
                        best_molecules.append(mol["smiles"])
                    elif isinstance(mol, str):
                        best_molecules.append(mol)
            else:
                best_molecules = valid_molecules[:10]

        # FIXED: Preserve order while removing duplicates
        def preserve_order_unique(lst):
            seen = set()
            result = []
            for item in lst:
                if item and item not in seen:  # Also filter out empty strings
                    seen.add(item)
                    result.append(item)
            return result

        all_generated = preserve_order_unique(all_generated)
        valid_molecules = preserve_order_unique(valid_molecules)
        invalid_molecules = preserve_order_unique(invalid_molecules)
        best_molecules = preserve_order_unique(best_molecules)

        return {
            "all_generated": all_generated,
            "valid_molecules": valid_molecules,
            "invalid_molecules": invalid_molecules,
            "total_generated": len(all_generated),
            "total_valid": len(valid_molecules),
            "total_invalid": len(invalid_molecules),
            "best_molecules": best_molecules,
            "generation_breakdown": generation_breakdown
        }

    def _safe_save_plot(self, fig, filepath, plot_name):
        """Safely save plot with error handling"""
        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save the plot
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✅ Saved {plot_name}: {filepath}")
            return True

        except Exception as e:
            print(f"⚠️ Failed to save {plot_name}: {e}")
            # Try alternative location
            try:
                alt_filepath = Path.cwd() / f"{plot_name}_{datetime.now().strftime('%H%M%S')}.png"
                fig.savefig(alt_filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                print(f"✅ Saved {plot_name} to alternative location: {alt_filepath}")
                return True
            except Exception as e2:
                print(f"❌ Failed to save {plot_name} to alternative location: {e2}")
                return False

    def analyze_and_visualize_results(self, experiment_results, save_plots=True, top_n=5):
        """Analyze results and create comprehensive visualizations"""
        query_name = experiment_results["query_name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Aggregate metrics across runs
        def aggregate_metrics(results_list):
            aggregated = {
                "validity": [],
                "uniqueness": [],
                "novelty": [],
                "diversity": [],
                "drug_likeness": [],
                "scaffold_diversity": [],
                "total_generated": [],
                "total_valid": [],
                "total_invalid": []
            }

            all_generated = []
            all_valid = []
            all_invalid = []
            all_best = []

            for run_data in results_list:
                if run_data.get("error"):
                    continue

                molecules_data = run_data.get("molecules_data", {})
                metrics = run_data.get("metrics", {})

                # Collect molecule counts
                aggregated["total_generated"].append(molecules_data.get("total_generated", 0))
                aggregated["total_valid"].append(molecules_data.get("total_valid", 0))
                aggregated["total_invalid"].append(molecules_data.get("total_invalid", 0))

                # Collect metrics
                aggregated["validity"].append(metrics.get("validity", {}).get("validity", 0))
                aggregated["uniqueness"].append(metrics.get("uniqueness", {}).get("uniqueness", 0))
                aggregated["novelty"].append(metrics.get("novelty", {}).get("novelty", 0))
                aggregated["diversity"].append(metrics.get("diversity", {}).get("diversity", 0))
                aggregated["drug_likeness"].append(metrics.get("drug_likeness", {}).get("drug_likeness", 0))
                aggregated["scaffold_diversity"].append(
                    metrics.get("scaffold_diversity", {}).get("scaffold_diversity", 0))

                # Collect all molecules
                all_generated.extend(molecules_data.get("all_generated", []))
                all_valid.extend(molecules_data.get("valid_molecules", []))
                all_invalid.extend(molecules_data.get("invalid_molecules", []))
                all_best.extend(molecules_data.get("best_molecules", []))

            # Calculate summary statistics
            summary_stats = {}
            for key, values in aggregated.items():
                if values:
                    summary_stats[key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "total": np.sum(values) if key.startswith("total_") else None,
                        "values": values
                    }

            return summary_stats, {
                "all_generated": list(set(all_generated)),
                "all_valid": list(set(all_valid)),
                "all_invalid": list(set(all_invalid)),
                "all_best": list(set(all_best))
            }

        # Aggregate results for both pipelines
        single_shot_stats, single_shot_molecules = aggregate_metrics(experiment_results["single_shot"])
        iterative_stats, iterative_molecules = aggregate_metrics(experiment_results["iterative"])

        # Create comprehensive metrics for visualization
        single_shot_comprehensive = self.metrics_calculator.comprehensive_evaluation(
            single_shot_molecules["all_generated"],
            [experiment_results["query_data"].get("target_smiles", "")]
        )

        iterative_comprehensive = self.metrics_calculator.comprehensive_evaluation(
            iterative_molecules["all_generated"],
            [experiment_results["query_data"].get("target_smiles", "")]
        )

        # Add molecule tracking to comprehensive results
        single_shot_comprehensive["molecule_tracking"] = {
            "total_generated": single_shot_stats.get("total_generated", {}).get("total", 0),
            "total_valid": single_shot_stats.get("total_valid", {}).get("total", 0),
            "total_invalid": single_shot_stats.get("total_invalid", {}).get("total", 0),
            "best_molecules": single_shot_molecules["all_best"][:5]
        }

        iterative_comprehensive["molecule_tracking"] = {
            "total_generated": iterative_stats.get("total_generated", {}).get("total", 0),
            "total_valid": iterative_stats.get("total_valid", {}).get("total", 0),
            "total_invalid": iterative_stats.get("total_invalid", {}).get("total", 0),
            "best_molecules": iterative_molecules["all_best"][:5]
        }

        single_shot_best = single_shot_molecules.get("all_best", [])
        iterative_best = iterative_molecules.get("all_best", [])

        # Get all generated molecules for internal diversity analysis
        single_shot_all = single_shot_molecules.get("all_generated", [])
        iterative_all = iterative_molecules.get("all_generated", [])

        # FIXED: Validate data before overlap analysis
        print(f"🔍 Overlap Analysis Debug:")
        print(f"Single-shot best: {len(single_shot_best)} molecules")
        print(f"Iterative best: {len(iterative_best)} molecules")
        print(f"Single-shot all: {len(single_shot_all)} molecules")
        print(f"Iterative all: {len(iterative_all)} molecules")

        # Calculate top-N overlap between single-shot and iterative best molecules
        top_n_overlap = self.metrics_calculator.calculate_top_n_overlap(
            single_shot_best, iterative_best, n=top_n
        )

        # Calculate positional overlap
        positional_overlap = self.metrics_calculator.calculate_positional_overlap(
            single_shot_best, iterative_best, n=top_n
        )

        # FIXED: Calculate multiple top-K values for comprehensive analysis
        top_k_analyses = {}
        for k in [1, 2, 3, 5, 10]:
            if k <= min(len(single_shot_best), len(iterative_best)):
                top_k_analyses[f"top_{k}"] = self.metrics_calculator.calculate_top_n_overlap(
                    single_shot_best, iterative_best, n=k
                )

        # Calculate internal diversity within each approach
        single_shot_internal_diversity = len(set(single_shot_all)) / len(single_shot_all) if single_shot_all else 0
        iterative_internal_diversity = len(set(iterative_all)) / len(iterative_all) if iterative_all else 0

        # FIXED: Comprehensive overlap analysis
        overlap_analysis = {
            "top_n_overlap": top_n_overlap,
            "positional_overlap": positional_overlap,
            "top_k_analyses": top_k_analyses,  # Multiple K values
            "internal_diversity": {
                "single_shot": single_shot_internal_diversity,
                "iterative": iterative_internal_diversity
            },
            "cross_approach_summary": {
                "total_unique_molecules": len(
                    set(single_shot_best + iterative_best)) if single_shot_best or iterative_best else 0,
                "single_shot_contributed": len(
                    set(single_shot_best) - set(iterative_best)) if single_shot_best and iterative_best else len(
                    single_shot_best),
                "iterative_contributed": len(
                    set(iterative_best) - set(single_shot_best)) if single_shot_best and iterative_best else len(
                    iterative_best),
                "shared_molecules": len(
                    set(single_shot_best) & set(iterative_best)) if single_shot_best and iterative_best else 0
            },
            "data_quality": {
                "single_shot_best_count": len(single_shot_best),
                "iterative_best_count": len(iterative_best),
                "single_shot_all_count": len(single_shot_all),
                "iterative_all_count": len(iterative_all),
                "sufficient_data": len(single_shot_best) >= top_n and len(iterative_best) >= top_n
            }
        }

        # Generate visualizations with safe saving
        if save_plots:
            print(f"🎨 Creating visualizations for {query_name}...")

            try:
                # Individual pipeline plots
                fig1 = self.visualizer.plot_comprehensive_metrics(
                    single_shot_comprehensive,
                    f"{query_name} - Single-shot Pipeline"
                )
                self._safe_save_plot(
                    fig1,
                    self.results_dir / f"{query_name}_single_shot_{timestamp}.png",
                    f"{query_name}_single_shot"
                )
                plt.close(fig1)

                fig2 = self.visualizer.plot_comprehensive_metrics(
                    iterative_comprehensive,
                    f"{query_name} - Iterative Pipeline"
                )
                self._safe_save_plot(
                    fig2,
                    self.results_dir / f"{query_name}_iterative_{timestamp}.png",
                    f"{query_name}_iterative"
                )
                plt.close(fig2)

                # Comparison plot
                fig3 = self.visualizer.plot_pipeline_comparison(
                    single_shot_comprehensive, iterative_comprehensive
                )
                self._safe_save_plot(
                    fig3,
                    self.results_dir / f"{query_name}_comparison_{timestamp}.png",
                    f"{query_name}_comparison"
                )
                plt.close(fig3)

                print(f"✅ All visualizations created for {query_name}")

                fig4 = self.visualizer.plot_top_n_overlap_analysis(
                    top_n_overlap,
                    f"{query_name} - Top-{top_n} Overlap Analysis (Single-shot vs Iterative)"
                )
                self._safe_save_plot(
                    fig4,
                    self.results_dir / f"{query_name}_top{top_n}_overlap_{timestamp}.png",
                    f"{query_name}_top{top_n}_overlap"
                )
                plt.close(fig4)

                print(f"✅ All visualizations created for {query_name}")
            except Exception as e:
                print(f"⚠️ Error creating visualizations for {query_name}: {e}")

            finally:
                plt.close('all')  # Clean up any remaining figures

        return {
            "query_name": query_name,
            "single_shot_stats": single_shot_stats,
            "iterative_stats": iterative_stats,
            "single_shot_comprehensive": single_shot_comprehensive,
            "iterative_comprehensive": iterative_comprehensive,
            "overlap_analysis": overlap_analysis,  # Enhanced overlap analysis
            "molecules": {
                "single_shot": single_shot_molecules,
                "iterative": iterative_molecules
            }
        }

    def run_comprehensive_experiment(self, query_names=None, runs=3, top_n=5):
        """Run comprehensive experiment with metrics and visualizations"""
        if query_names is None:
            query_names = get_query_list()[:5]  # First 5 queries as requested

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []
        all_analyses = []

        print(f"🚀 Starting comprehensive experiment with {len(query_names)} queries")
        print(f"📋 Queries: {query_names}")
        print(f"🔄 Runs per query: {runs}")
        print(f"⏰ Timestamp: {timestamp}")
        print(f"📁 Results directory: {self.results_dir.absolute()}")

        for i, query_name in enumerate(query_names):
            print(f"\n{'=' * 60}")
            print(f"Processing query {i + 1}/{len(query_names)}: {query_name}")
            print(f"{'=' * 60}")

            try:
                query_data = get_query_data(query_name)

                # Run experiment
                experiment_results = self.run_single_query_experiment(
                    query_name, query_data, runs=runs
                )

                # Analyze and visualize
                analysis = self.analyze_and_visualize_results(experiment_results, top_n=top_n)
                all_results.append(experiment_results)
                all_analyses.append(analysis)

                # Save individual results with safe file handling
                try:
                    result_file = self.results_dir / f"{query_name}_detailed_{timestamp}.json"
                    result_file.parent.mkdir(parents=True, exist_ok=True)

                    with open(result_file, 'w') as f:
                        json.dump(experiment_results, f, indent=2, default=str)
                    print(f"✅ Saved detailed results: {result_file}")

                except Exception as e:
                    print(f"⚠️ Failed to save detailed results for {query_name}: {e}")

                # Print summary
                single_comprehensive = analysis["single_shot_comprehensive"]
                iterative_comprehensive = analysis["iterative_comprehensive"]

                print(f"\n📊 Results Summary for {query_name}:")
                print(f"Single-shot - Generated: {single_comprehensive['molecule_tracking']['total_generated']}, "
                      f"Valid: {single_comprehensive['molecule_tracking']['total_valid']}, "
                      f"Invalid: {single_comprehensive['molecule_tracking']['total_invalid']}")
                print(f"Iterative   - Generated: {iterative_comprehensive['molecule_tracking']['total_generated']}, "
                      f"Valid: {iterative_comprehensive['molecule_tracking']['total_valid']}, "
                      f"Invalid: {iterative_comprehensive['molecule_tracking']['total_invalid']}")

            except Exception as e:
                print(f"❌ Failed to process query {query_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save comprehensive summary with error handling
        try:
            summary_data = {
                "analyses": all_analyses,
                "metadata": {
                    "queries": query_names,
                    "runs_per_query": runs,
                    "timestamp": timestamp,
                    "total_experiments": len(query_names) * runs * 2  # 2 pipelines
                }
            }

            summary_file = self.results_dir / f"comprehensive_summary_{timestamp}.json"
            summary_file.parent.mkdir(parents=True, exist_ok=True)

            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            print(f"✅ Saved comprehensive summary: {summary_file}")

        except Exception as e:
            print(f"⚠️ Failed to save comprehensive summary: {e}")

        # Generate final summary table
        if all_analyses:
            try:
                self.generate_comprehensive_summary_table(all_analyses, timestamp)
            except Exception as e:
                print(f"⚠️ Failed to generate summary table: {e}")

        return all_results, all_analyses

    def generate_comprehensive_summary_table(self, analyses, timestamp):
        """Generate comprehensive summary table with molecular metrics"""
        summary_lines = []
        summary_lines.append(f"Comprehensive Molecular Generation Experiment - {timestamp}")
        summary_lines.append("=" * 120)
        summary_lines.append("")

        # Headers
        header = f"{'Query':<20} {'Pipeline':<12} {'Generated':<9} {'Valid':<6} {'Invalid':<7} {'Unique':<6} {'Novel':<6} {'Diverse':<7} {'DrugLike':<8} {'Success':<7}"
        summary_lines.append(header)
        summary_lines.append("-" * 120)

        # Collect statistics for final summary
        single_shot_metrics = {
            'success': [], 'validity': [], 'uniqueness': [],
            'novelty': [], 'diversity': [], 'drug_likeness': [],
            'total_generated': [], 'total_valid': [], 'total_invalid': []
        }
        iterative_metrics = {
            'success': [], 'validity': [], 'uniqueness': [],
            'novelty': [], 'diversity': [], 'drug_likeness': [],
            'total_generated': [], 'total_valid': [], 'total_invalid': []
        }

        for analysis in analyses:
            query_name = analysis["query_name"][:19]

            # Single-shot metrics
            ss = analysis["single_shot_comprehensive"]
            ss_tracking = ss.get("molecule_tracking", {})
            ss_line = f"{query_name:<20} {'Single':<12} "
            ss_line += f"{ss_tracking.get('total_generated', 0):<9} "
            ss_line += f"{ss_tracking.get('total_valid', 0):<6} "
            ss_line += f"{ss_tracking.get('total_invalid', 0):<7} "
            ss_line += f"{ss['uniqueness']['uniqueness']:<6.3f} "
            ss_line += f"{ss.get('novelty', {}).get('novelty', 0):<6.3f} "
            ss_line += f"{ss['diversity']['diversity']:<7.3f} "
            ss_line += f"{ss['drug_likeness']['drug_likeness']:<8.3f} "
            ss_line += f"{ss['summary']['overall_success_rate']:<7.3f}"
            summary_lines.append(ss_line)

            # Iterative metrics
            it = analysis["iterative_comprehensive"]
            it_tracking = it.get("molecule_tracking", {})
            it_line = f"{'':<20} {'Iterative':<12} "
            it_line += f"{it_tracking.get('total_generated', 0):<9} "
            it_line += f"{it_tracking.get('total_valid', 0):<6} "
            it_line += f"{it_tracking.get('total_invalid', 0):<7} "
            it_line += f"{it['uniqueness']['uniqueness']:<6.3f} "
            it_line += f"{it.get('novelty', {}).get('novelty', 0):<6.3f} "
            it_line += f"{it['diversity']['diversity']:<7.3f} "
            it_line += f"{it['drug_likeness']['drug_likeness']:<8.3f} "
            it_line += f"{it['summary']['overall_success_rate']:<7.3f}"
            summary_lines.append(it_line)
            summary_lines.append("")

            # Collect for statistics
            single_shot_metrics['success'].append(ss['summary']['overall_success_rate'])
            single_shot_metrics['validity'].append(ss['validity']['validity'])
            single_shot_metrics['uniqueness'].append(ss['uniqueness']['uniqueness'])
            single_shot_metrics['novelty'].append(ss.get('novelty', {}).get('novelty', 0))
            single_shot_metrics['diversity'].append(ss['diversity']['diversity'])
            single_shot_metrics['drug_likeness'].append(ss['drug_likeness']['drug_likeness'])
            single_shot_metrics['total_generated'].append(ss_tracking.get('total_generated', 0))
            single_shot_metrics['total_valid'].append(ss_tracking.get('total_valid', 0))
            single_shot_metrics['total_invalid'].append(ss_tracking.get('total_invalid', 0))

            iterative_metrics['success'].append(it['summary']['overall_success_rate'])
            iterative_metrics['validity'].append(it['validity']['validity'])
            iterative_metrics['uniqueness'].append(it['uniqueness']['uniqueness'])
            iterative_metrics['novelty'].append(it.get('novelty', {}).get('novelty', 0))
            iterative_metrics['diversity'].append(it['diversity']['diversity'])
            iterative_metrics['drug_likeness'].append(it['drug_likeness']['drug_likeness'])
            iterative_metrics['total_generated'].append(it_tracking.get('total_generated', 0))
            iterative_metrics['total_valid'].append(it_tracking.get('total_valid', 0))
            iterative_metrics['total_invalid'].append(it_tracking.get('total_invalid', 0))

        # Overall statistics
        summary_lines.append("=" * 120)
        summary_lines.append("OVERALL STATISTICS")
        summary_lines.append("=" * 120)

        metrics_names = ['Generated', 'Valid', 'Invalid', 'Success', 'Validity', 'Uniqueness', 'Novelty', 'Diversity',
                         'Drug-likeness']

        for metric_name, metric_key in zip(metrics_names, single_shot_metrics.keys()):
            ss_values = single_shot_metrics[metric_key]
            it_values = iterative_metrics[metric_key]

            if ss_values and it_values:
                if metric_key.startswith('total_'):
                    summary_lines.append(f"{metric_name}:")
                    summary_lines.append(
                        f"  Single-shot: Total={sum(ss_values)}, Mean={np.mean(ss_values):.1f} ± {np.std(ss_values):.1f}")
                    summary_lines.append(
                        f"  Iterative:   Total={sum(it_values)}, Mean={np.mean(it_values):.1f} ± {np.std(it_values):.1f}")
                else:
                    summary_lines.append(f"{metric_name}:")
                    summary_lines.append(f"  Single-shot: Mean={np.mean(ss_values):.3f} ± {np.std(ss_values):.3f}")
                    summary_lines.append(f"  Iterative:   Mean={np.mean(it_values):.3f} ± {np.std(it_values):.3f}")
                summary_lines.append("")

        # Best molecules section
        summary_lines.append("=" * 120)
        summary_lines.append("BEST MOLECULES")
        summary_lines.append("=" * 120)

        for analysis in analyses:
            query_name = analysis["query_name"]
            summary_lines.append(f"\n{query_name}:")

            ss_best = analysis["single_shot_comprehensive"]["molecule_tracking"]["best_molecules"]
            it_best = analysis["iterative_comprehensive"]["molecule_tracking"]["best_molecules"]

            summary_lines.append(f"  Single-shot best: {ss_best[:3]}")
            summary_lines.append(f"  Iterative best:   {it_best[:3]}")

        # Save summary with error handling
        try:
            summary_file = self.results_dir / f"comprehensive_table_{timestamp}.txt"
            summary_file.parent.mkdir(parents=True, exist_ok=True)

            with open(summary_file, 'w') as f:
                f.write('\n'.join(summary_lines))
            print(f"✅ Saved summary table: {summary_file}")

        except Exception as e:
            print(f"⚠️ Failed to save summary table: {e}")
            # Try saving to current directory
            try:
                alt_file = Path.cwd() / f"summary_table_{timestamp}.txt"
                with open(alt_file, 'w') as f:
                    f.write('\n'.join(summary_lines))
                print(f"✅ Saved summary table to alternative location: {alt_file}")
            except Exception as e2:
                print(f"❌ Failed to save summary table anywhere: {e2}")

        # Print to console
        print("\n" + '\n'.join(summary_lines))


if __name__ == "__main__":
    experiment = ExperimentOne()

    test_queries =get_query_list()

    try:
        print("🚀 Starting comprehensive molecular generation experiment...")
        results, analyses = experiment.run_comprehensive_experiment(
            query_names=test_queries,
            runs=3
        )
        print("\n🎉 Comprehensive experiment completed successfully!")
        print(f"📊 Results saved in: {experiment.results_dir}")
        print(f"📈 Visualizations generated for each query")

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback

        traceback.print_exc()