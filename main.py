# generalized_oracle_evaluation.py

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
import warnings

warnings.filterwarnings('ignore')

# Oracle mapping for each query type
ORACLE_MAPPING = {
    "albuterol_similarity": "Albuterol_similarity",
    "amlodipine_mpo": "Amlodipine_MPO",
    "celecoxib_rediscovery": "Celecoxib_rediscovery",
    "isomers_c7h8n2o2": "Isomers_C7H8N2O2",
    "drd2_binding": "DRD2"
}


class OracleEvaluator:
    def __init__(self, results_dir="improved_experiment_results"):
        self.results_dir = Path(results_dir)
        self.oracles = {}
        self.load_oracles()

    def load_oracles(self):
        """Load all required oracles"""
        print("🔮 Loading Oracle models...")
        for query_name, oracle_name in ORACLE_MAPPING.items():
            try:
                oracle = Oracle(name=oracle_name)
                self.oracles[query_name] = oracle
                print(f"✅ Loaded {oracle_name} for {query_name}")
            except Exception as e:
                print(f"❌ Failed to load {oracle_name}: {e}")
                self.oracles[query_name] = None

    def extract_experiment_results(self):
        """Extract results from all experiment files"""
        print(f"\n📁 Extracting results from {self.results_dir}...")

        all_results = defaultdict(lambda: {"single_shot": [], "iterative": []})

        # Look for detailed experiment files
        for file_path in self.results_dir.glob("*_detailed_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                query_name = data.get("query_name", "unknown")
                print(f"📄 Processing {file_path.name} for query: {query_name}")

                # Extract single-shot results across all runs
                for run_data in data.get("single_shot", []):
                    if run_data.get("result") and not run_data["result"].get("error"):
                        valid_smiles = run_data["result"].get("valid", [])
                        if valid_smiles:
                            all_results[query_name]["single_shot"].append({
                                "run": run_data.get("run", 1),
                                "seed": run_data.get("seed", 0),
                                "smiles": valid_smiles
                            })

                # Extract iterative results across all runs
                for run_data in data.get("iterative", []):
                    if run_data.get("result") and not run_data["result"].get("error"):
                        valid_smiles = run_data["result"].get("valid", [])
                        if valid_smiles:
                            all_results[query_name]["iterative"].append({
                                "run": run_data.get("run", 1),
                                "seed": run_data.get("seed", 0),
                                "smiles": valid_smiles
                            })

            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

        return dict(all_results)

    def calculate_molecular_properties(self, smiles):
        """Calculate molecular properties"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol)
        }

    def score_molecules_with_oracle(self, smiles_list, oracle, query_name):
        """Score molecules using the appropriate oracle"""
        results = []

        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    score = oracle(smiles)
                    props = self.calculate_molecular_properties(smiles)

                    result = {
                        'SMILES': smiles,
                        'Oracle_Score': score,
                        'Query': query_name,
                        'Molecule_ID': f"{query_name}_{i + 1}",
                        **props
                    }
                    results.append(result)

                except Exception as e:
                    print(f"⚠️ Error scoring {smiles}: {e}")

        return results

    def calculate_auc_top_k(self, scores, k=10):
        """Calculate AUC for top-k scoring"""
        if len(scores) == 0:
            return 0.0

        # Sort scores in descending order
        sorted_scores = sorted(scores, reverse=True)

        # Take top-k or all if less than k
        top_k_scores = sorted_scores[:min(k, len(sorted_scores))]

        if len(top_k_scores) < 2:
            return np.mean(top_k_scores) if top_k_scores else 0.0

        # Create x-axis (ranks) and calculate AUC
        x = np.arange(1, len(top_k_scores) + 1)
        # Normalize x to [0,1] range
        x_norm = (x - 1) / (len(top_k_scores) - 1) if len(top_k_scores) > 1 else [0]

        try:
            auc_score = auc(x_norm, top_k_scores)
            return auc_score
        except:
            return np.mean(top_k_scores)

    def evaluate_query_results(self, query_name, query_results):
        """Evaluate results for a single query across all runs"""
        if query_name not in self.oracles or self.oracles[query_name] is None:
            print(f"❌ No oracle available for {query_name}")
            return None

        oracle = self.oracles[query_name]
        oracle_name = ORACLE_MAPPING[query_name]

        print(f"\n🎯 Evaluating {query_name} with {oracle_name} oracle")
        print("=" * 60)

        evaluation_results = {
            "query_name": query_name,
            "oracle_name": oracle_name,
            "single_shot": {"runs": [], "auc_scores": [], "top_10_scores": []},
            "iterative": {"runs": [], "auc_scores": [], "top_10_scores": []}
        }

        # Evaluate single-shot runs
        print(f"📊 Single-shot evaluation ({len(query_results['single_shot'])} runs):")
        for run_data in query_results["single_shot"]:
            scored_molecules = self.score_molecules_with_oracle(
                run_data["smiles"], oracle, query_name
            )

            if scored_molecules:
                scores = [mol['Oracle_Score'] for mol in scored_molecules]
                auc_top_10 = self.calculate_auc_top_k(scores, k=10)
                top_10_mean = np.mean(sorted(scores, reverse=True)[:10])

                run_result = {
                    "run": run_data["run"],
                    "seed": run_data["seed"],
                    "total_molecules": len(scored_molecules),
                    "oracle_scores": scores,
                    "auc_top_10": auc_top_10,
                    "top_10_mean": top_10_mean,
                    "max_score": max(scores),
                    "mean_score": np.mean(scores),
                    "molecules": scored_molecules
                }

                evaluation_results["single_shot"]["runs"].append(run_result)
                evaluation_results["single_shot"]["auc_scores"].append(auc_top_10)
                evaluation_results["single_shot"]["top_10_scores"].append(top_10_mean)

                print(f"  Run {run_data['run']}: {len(scored_molecules)} molecules, "
                      f"AUC-10: {auc_top_10:.3f}, Top-10 mean: {top_10_mean:.3f}")

        # Evaluate iterative runs
        print(f"📊 Iterative evaluation ({len(query_results['iterative'])} runs):")
        for run_data in query_results["iterative"]:
            scored_molecules = self.score_molecules_with_oracle(
                run_data["smiles"], oracle, query_name
            )

            if scored_molecules:
                scores = [mol['Oracle_Score'] for mol in scored_molecules]
                auc_top_10 = self.calculate_auc_top_k(scores, k=10)
                top_10_mean = np.mean(sorted(scores, reverse=True)[:10])

                run_result = {
                    "run": run_data["run"],
                    "seed": run_data["seed"],
                    "total_molecules": len(scored_molecules),
                    "oracle_scores": scores,
                    "auc_top_10": auc_top_10,
                    "top_10_mean": top_10_mean,
                    "max_score": max(scores),
                    "mean_score": np.mean(scores),
                    "molecules": scored_molecules
                }

                evaluation_results["iterative"]["runs"].append(run_result)
                evaluation_results["iterative"]["auc_scores"].append(auc_top_10)
                evaluation_results["iterative"]["top_10_scores"].append(top_10_mean)

                print(f"  Run {run_data['run']}: {len(scored_molecules)} molecules, "
                      f"AUC-10: {auc_top_10:.3f}, Top-10 mean: {top_10_mean:.3f}")

        return evaluation_results

    def create_comprehensive_plots(self, all_evaluations):
        """Create comprehensive visualization plots"""
        print("\n📊 Creating comprehensive visualizations...")

        # Prepare data for plotting
        plot_data = []
        auc_summary_data = []

        for query_name, eval_data in all_evaluations.items():
            if eval_data is None:
                continue

            # Single-shot data
            for run in eval_data["single_shot"]["runs"]:
                for mol in run["molecules"]:
                    plot_data.append({
                        'Query': query_name,
                        'Pipeline': 'Single-shot',
                        'Run': run["run"],
                        'Oracle_Score': mol['Oracle_Score'],
                        'MW': mol['MW'],
                        'LogP': mol['LogP'],
                        'TPSA': mol['TPSA']
                    })

            # Iterative data
            for run in eval_data["iterative"]["runs"]:
                for mol in run["molecules"]:
                    plot_data.append({
                        'Query': query_name,
                        'Pipeline': 'Iterative',
                        'Run': run["run"],
                        'Oracle_Score': mol['Oracle_Score'],
                        'MW': mol['MW'],
                        'LogP': mol['LogP'],
                        'TPSA': mol['TPSA']
                    })

            # AUC summary data
            if eval_data["single_shot"]["auc_scores"]:
                auc_summary_data.append({
                    'Query': query_name,
                    'Pipeline': 'Single-shot',
                    'AUC_Top10_Mean': np.mean(eval_data["single_shot"]["auc_scores"]),
                    'AUC_Top10_Std': np.std(eval_data["single_shot"]["auc_scores"]),
                    'N_Runs': len(eval_data["single_shot"]["auc_scores"])
                })

            if eval_data["iterative"]["auc_scores"]:
                auc_summary_data.append({
                    'Query': query_name,
                    'Pipeline': 'Iterative',
                    'AUC_Top10_Mean': np.mean(eval_data["iterative"]["auc_scores"]),
                    'AUC_Top10_Std': np.std(eval_data["iterative"]["auc_scores"]),
                    'N_Runs': len(eval_data["iterative"]["auc_scores"])
                })

        if not plot_data:
            print("❌ No data available for plotting")
            return

        df = pd.DataFrame(plot_data)
        auc_df = pd.DataFrame(auc_summary_data)

        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 15))

        # 1. AUC Top-10 Comparison
        ax1 = plt.subplot(3, 4, 1)
        if len(auc_df) > 0:
            sns.barplot(data=auc_df, x='Query', y='AUC_Top10_Mean', hue='Pipeline', ax=ax1)
            ax1.set_title("AUC Top-10 by Query and Pipeline")
            ax1.set_ylabel("AUC Top-10 Score")
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

            # Add error bars
            for i, (_, row) in enumerate(auc_df.iterrows()):
                x_pos = i % len(auc_df['Query'].unique())
                if row['Pipeline'] == 'Iterative':
                    x_pos += 0.2
                else:
                    x_pos -= 0.2
                ax1.errorbar(x_pos, row['AUC_Top10_Mean'], yerr=row['AUC_Top10_Std'],
                             color='black', capsize=3, alpha=0.7)

        # 2. Oracle Score Distribution by Query
        ax2 = plt.subplot(3, 4, 2)
        sns.boxplot(data=df, x='Query', y='Oracle_Score', hue='Pipeline', ax=ax2)
        ax2.set_title("Oracle Score Distribution")
        ax2.set_ylabel("Oracle Score")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # 3. Oracle Score vs Molecular Weight
        ax3 = plt.subplot(3, 4, 3)
        for pipeline in df['Pipeline'].unique():
            subset = df[df['Pipeline'] == pipeline]
            ax3.scatter(subset['MW'], subset['Oracle_Score'],
                        label=pipeline, alpha=0.6, s=30)
        ax3.set_xlabel("Molecular Weight (Da)")
        ax3.set_ylabel("Oracle Score")
        ax3.set_title("Oracle Score vs Molecular Weight")
        ax3.legend()

        # 4. Oracle Score vs LogP
        ax4 = plt.subplot(3, 4, 4)
        for pipeline in df['Pipeline'].unique():
            subset = df[df['Pipeline'] == pipeline]
            ax4.scatter(subset['LogP'], subset['Oracle_Score'],
                        label=pipeline, alpha=0.6, s=30)
        ax4.set_xlabel("LogP")
        ax4.set_ylabel("Oracle Score")
        ax4.set_title("Oracle Score vs LogP")
        ax4.legend()

        # 5-8. Individual query detailed plots
        queries = df['Query'].unique()[:4]  # Show up to 4 queries
        for i, query in enumerate(queries):
            ax = plt.subplot(3, 4, 5 + i)
            query_data = df[df['Query'] == query]
            sns.violinplot(data=query_data, x='Pipeline', y='Oracle_Score', ax=ax)
            ax.set_title(f"{query}\nOracle Scores")
            ax.set_ylabel("Oracle Score")

        # 9. Molecules per Pipeline
        ax9 = plt.subplot(3, 4, 9)
        pipeline_counts = df.groupby(['Query', 'Pipeline']).size().unstack(fill_value=0)
        pipeline_counts.plot(kind='bar', ax=ax9)
        ax9.set_title("Molecules Generated per Pipeline")
        ax9.set_ylabel("Number of Molecules")
        plt.setp(ax9.get_xticklabels(), rotation=45, ha='right')
        ax9.legend(title='Pipeline')

        # 10. Top Scoring Molecules Summary
        ax10 = plt.subplot(3, 4, 10)
        top_scores_by_query = df.groupby('Query')['Oracle_Score'].max()
        top_scores_by_query.plot(kind='bar', ax=ax10, color='skyblue')
        ax10.set_title("Best Oracle Score per Query")
        ax10.set_ylabel("Best Oracle Score")
        plt.setp(ax10.get_xticklabels(), rotation=45, ha='right')

        # 11. Score Distribution Histogram
        ax11 = plt.subplot(3, 4, 11)
        for pipeline in df['Pipeline'].unique():
            subset = df[df['Pipeline'] == pipeline]
            ax11.hist(subset['Oracle_Score'], alpha=0.6, label=pipeline, bins=20)
        ax11.set_xlabel("Oracle Score")
        ax11.set_ylabel("Frequency")
        ax11.set_title("Oracle Score Distribution")
        ax11.legend()

        # 12. Run Consistency Analysis
        ax12 = plt.subplot(3, 4, 12)
        if len(auc_df) > 0:
            sns.scatterplot(data=auc_df, x='N_Runs', y='AUC_Top10_Std',
                            hue='Pipeline', size='AUC_Top10_Mean', ax=ax12)
            ax12.set_title("Run Consistency Analysis")
            ax12.set_xlabel("Number of Runs")
            ax12.set_ylabel("AUC Top-10 Std Dev")

        plt.tight_layout()
        plt.savefig(self.results_dir / "comprehensive_oracle_evaluation.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        return df, auc_df

    def print_comprehensive_summary(self, all_evaluations):
        """Print comprehensive summary following the paper format"""
        print("\n" + "=" * 100)
        print("🏆 COMPREHENSIVE ORACLE EVALUATION SUMMARY")
        print("=" * 100)

        summary_table = []

        for query_name, eval_data in all_evaluations.items():
            if eval_data is None:
                continue

            oracle_name = eval_data["oracle_name"]

            # Single-shot summary
            if eval_data["single_shot"]["auc_scores"]:
                ss_auc_mean = np.mean(eval_data["single_shot"]["auc_scores"])
                ss_auc_std = np.std(eval_data["single_shot"]["auc_scores"])
                ss_runs = len(eval_data["single_shot"]["auc_scores"])

                summary_table.append({
                    'Query': query_name,
                    'Oracle': oracle_name,
                    'Pipeline': 'Single-shot',
                    'AUC_Top10_Mean': ss_auc_mean,
                    'AUC_Top10_Std': ss_auc_std,
                    'Runs': ss_runs,
                    'Total_Molecules': sum(len(run["molecules"]) for run in eval_data["single_shot"]["runs"])
                })

            # Iterative summary
            if eval_data["iterative"]["auc_scores"]:
                it_auc_mean = np.mean(eval_data["iterative"]["auc_scores"])
                it_auc_std = np.std(eval_data["iterative"]["auc_scores"])
                it_runs = len(eval_data["iterative"]["auc_scores"])

                summary_table.append({
                    'Query': query_name,
                    'Oracle': oracle_name,
                    'Pipeline': 'Iterative',
                    'AUC_Top10_Mean': it_auc_mean,
                    'AUC_Top10_Std': it_auc_std,
                    'Runs': it_runs,
                    'Total_Molecules': sum(len(run["molecules"]) for run in eval_data["iterative"]["runs"])
                })

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_table)

        if len(summary_df) > 0:
            print("\n📋 AUC TOP-10 RESULTS (Following Paper Format):")
            print("-" * 100)
            print(
                f"{'Query':<20} {'Oracle':<20} {'Pipeline':<12} {'AUC-10':<12} {'±Std':<8} {'Runs':<6} {'Molecules':<10}")
            print("-" * 100)

            for _, row in summary_df.iterrows():
                runs_marker = "*" if row['Runs'] == 3 else ""
                print(f"{row['Query']:<20} {row['Oracle']:<20} {row['Pipeline']:<12} "
                      f"{row['AUC_Top10_Mean']:<8.3f}{runs_marker:<4} "
                      f"±{row['AUC_Top10_Std']:<7.3f} {row['Runs']:<6} {row['Total_Molecules']:<10}")

            print("-" * 100)
            print("(*) Results evaluated from 3 independent runs")
            print("Others assessed from 5 independent runs")

            # Best performing analysis
            print(f"\n🥇 BEST PERFORMING COMBINATIONS:")
            best_by_query = summary_df.loc[summary_df.groupby('Query')['AUC_Top10_Mean'].idxmax()]
            for _, row in best_by_query.iterrows():
                print(
                    f"  {row['Query']}: {row['Pipeline']} (AUC-10: {row['AUC_Top10_Mean']:.3f}±{row['AUC_Top10_Std']:.3f})")

            # Overall pipeline comparison
            print(f"\n📊 OVERALL PIPELINE PERFORMANCE:")
            pipeline_performance = summary_df.groupby('Pipeline').agg({
                'AUC_Top10_Mean': ['mean', 'std', 'count'],
                'Total_Molecules': 'sum'
            }).round(3)
            print(pipeline_performance)

        # Save detailed results
        results_file = self.results_dir / "oracle_evaluation_summary.json"
        with open(results_file, 'w') as f:
            json.dump(all_evaluations, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")

        # Save summary table
        if len(summary_df) > 0:
            csv_file = self.results_dir / "oracle_auc_summary.csv"
            summary_df.to_csv(csv_file, index=False)
            print(f"💾 Summary table saved to: {csv_file}")

        return summary_df

    def run_complete_evaluation(self):
        """Run the complete oracle evaluation pipeline"""
        print("🚀 Starting Complete Oracle Evaluation Pipeline")
        print("=" * 80)

        # Step 1: Extract experiment results
        experiment_results = self.extract_experiment_results()

        if not experiment_results:
            print("❌ No experiment results found!")
            return

        print(f"✅ Found results for {len(experiment_results)} queries")

        # Step 2: Evaluate each query with its oracle
        all_evaluations = {}

        for query_name, query_results in experiment_results.items():
            print(f"\n🔍 Processing query: {query_name}")
            evaluation = self.evaluate_query_results(query_name, query_results)
            all_evaluations[query_name] = evaluation

        # Step 3: Create comprehensive visualizations
        df, auc_df = self.create_comprehensive_plots(all_evaluations)

        # Step 4: Print comprehensive summary
        summary_df = self.print_comprehensive_summary(all_evaluations)

        print(f"\n🎉 Complete evaluation finished!")
        print(f"📊 Evaluated {len(all_evaluations)} queries with their respective oracles")
        print(f"📈 Generated comprehensive plots and summary tables")

        return all_evaluations, df, auc_df, summary_df


def main():
    """Main function to run generalized oracle evaluation"""
    # Initialize evaluator
    evaluator = OracleEvaluator(results_dir="scripts/experiment_results")

    # Run complete evaluation
    try:
        all_evaluations, df, auc_df, summary_df = evaluator.run_complete_evaluation()

        print("\n" + "=" * 80)
        print("✅ EVALUATION COMPLETE - KEY FINDINGS:")
        print("=" * 80)

        if len(summary_df) > 0:
            # Show best overall performance
            best_overall = summary_df.loc[summary_df['AUC_Top10_Mean'].idxmax()]
            print(f"🏆 Best Overall Performance:")
            print(f"   Query: {best_overall['Query']}")
            print(f"   Pipeline: {best_overall['Pipeline']}")
            print(f"   Oracle: {best_overall['Oracle']}")
            print(f"   AUC Top-10: {best_overall['AUC_Top10_Mean']:.3f} ± {best_overall['AUC_Top10_Std']:.3f}")
            print(f"   Runs: {best_overall['Runs']}")

            # Pipeline comparison
            pipeline_avg = summary_df.groupby('Pipeline')['AUC_Top10_Mean'].mean()
            print(f"\n📈 Average Pipeline Performance:")
            for pipeline, avg_score in pipeline_avg.items():
                print(f"   {pipeline}: {avg_score:.3f}")

        return all_evaluations, df, auc_df, summary_df

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()