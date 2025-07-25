# corrected_oracle_evaluation.py

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED
from sklearn.metrics import auc
import warnings
from rdkit import RDLogger

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# COMPLETE ORACLE MAPPING - ALL AVAILABLE IN TDC
COMPLETE_ORACLE_MAPPING = {
    "albuterol_similarity": "Albuterol_Similarity",
    "amlodipine_mpo": "Amlodipine_MPO",
    "celecoxib_rediscovery": "Celecoxib_Rediscovery",
    "deco_hop": "Deco Hop",
    "drd2_binding": "DRD2",
    "fexofenadine_mpo": "Fexofenadine_MPO",
    "gsk3b_activity": "GSK3B",
    "isomers_c7h8n2o2": "Isomers_C7H8N2O2",
    "isomers_c9h10n2o2pf2cl": "Isomers_C9H10N2O2PF2Cl",
    "jnk3_inhibition": "JNK3",
    "median1_similarity": "Median 1",
    "median2_similarity": "Median 2",
    "mestranol_similarity": "Mestranol_Similarity",
    "osimertinib_mpo": "Osimertinib_MPO",
    "perindopril_mpo": "Perindopril_MPO",
    "qed_optimization": "QED",
    "ranolazine_mpo": "Ranolazine_MPO",
    "scaffold_hop": "Scaffold Hop",
    "sitagliptin_mpo": "Sitagliptin_MPO",
    "thiothixene_similarity": "Thiothixene_Rediscovery",
    "troglitazone_similarity": "Troglitazone_Rediscovery",
    "valsartan_smarts": "Valsartan_SMARTS",
    "zaleplon_similarity": "Zaleplon_MPO"
}


class ComprehensiveOracleEvaluator:
    def __init__(self, results_dir="improved_experiment_results"):
        self.results_dir = Path(results_dir)
        self.oracles = {}
        self.load_all_oracles()

    def load_all_oracles(self):
        """Load all TDC oracles"""
        print("🔮 Loading TDC Oracle models...")

        for query_name, oracle_name in COMPLETE_ORACLE_MAPPING.items():
            try:
                oracle = Oracle(name=oracle_name)
                self.oracles[query_name] = oracle
                print(f"✅ Loaded {oracle_name} for {query_name}")
            except Exception as e:
                print(f"❌ Failed to load {oracle_name}: {e}")
                self.oracles[query_name] = None

        print(
            f"\n📊 Successfully loaded {len([o for o in self.oracles.values() if o is not None])}/{len(COMPLETE_ORACLE_MAPPING)} oracles")

    def score_molecule(self, smiles, query_name):
        """Score a single molecule using TDC oracle"""
        if query_name not in self.oracles or self.oracles[query_name] is None:
            return 0.0

        try:
            score = self.oracles[query_name](smiles)
            return float(score) if score is not None else 0.0
        except Exception as e:
            print(f"⚠️ Oracle scoring failed for {smiles}: {e}")
            return 0.0

    def calculate_molecular_properties(self, smiles):
        """Calculate molecular properties"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        return {
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
            'QED': QED.qed(mol)
        }

    def calculate_auc_top_k(self, scores, k=10):
        """Calculate AUC for top-k molecules"""
        if len(scores) == 0:
            return 0.0

        # Sort scores in descending order
        sorted_scores = sorted(scores, reverse=True)
        top_k_scores = sorted_scores[:min(k, len(sorted_scores))]

        if len(top_k_scores) < 2:
            return np.mean(top_k_scores) if top_k_scores else 0.0

        # Create normalized x-axis (ranks from 0 to 1)
        x = np.linspace(0, 1, len(top_k_scores))

        try:
            auc_score = auc(x, top_k_scores)
            return auc_score
        except:
            return np.mean(top_k_scores)

    def extract_experiment_results(self):
        """Extract results from experiment files"""
        print(f"\n📁 Extracting results from {self.results_dir}...")

        all_results = defaultdict(lambda: {"single_shot": [], "iterative": []})

        for file_path in self.results_dir.glob("*_detailed_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                query_name = data.get("query_name", "unknown")
                print(f"📄 Processing {file_path.name} for query: {query_name}")

                # Extract single-shot results
                for run_data in data.get("single_shot", []):
                    if run_data.get("result") and not run_data["result"].get("error"):
                        valid_smiles = run_data["result"].get("valid", [])
                        if valid_smiles:
                            all_results[query_name]["single_shot"].append({
                                "run": run_data.get("run", 1),
                                "seed": run_data.get("seed", 0),
                                "smiles": valid_smiles
                            })

                # Extract iterative results
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

    def evaluate_query_results(self, query_name, query_results):
        """Evaluate results for a single query"""
        if query_name not in self.oracles or self.oracles[query_name] is None:
            print(f"❌ No oracle available for {query_name}")
            return None

        oracle_name = COMPLETE_ORACLE_MAPPING[query_name]
        print(f"\n🎯 Evaluating {query_name} with {oracle_name}")
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
            scored_molecules = []

            for i, smiles in enumerate(run_data["smiles"]):
                score = self.score_molecule(smiles, query_name)
                props = self.calculate_molecular_properties(smiles)

                molecule_result = {
                    'SMILES': smiles,
                    'Oracle_Score': score,
                    'Query': query_name,
                    'Molecule_ID': f"{query_name}_ss_{run_data['run']}_{i + 1}",
                    **props
                }
                scored_molecules.append(molecule_result)

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
                    "max_score": max(scores) if scores else 0,
                    "mean_score": np.mean(scores) if scores else 0,
                    "molecules": scored_molecules
                }

                evaluation_results["single_shot"]["runs"].append(run_result)
                evaluation_results["single_shot"]["auc_scores"].append(auc_top_10)
                evaluation_results["single_shot"]["top_10_scores"].append(top_10_mean)

                print(f"  Run {run_data['run']}: {len(scored_molecules)} molecules, "
                      f"AUC-10: {auc_top_10:.4f}, Top-10: {top_10_mean:.4f}, "
                      f"Max: {max(scores):.4f}")

        # Evaluate iterative runs
        print(f"📊 Iterative evaluation ({len(query_results['iterative'])} runs):")
        for run_data in query_results["iterative"]:
            scored_molecules = []

            for i, smiles in enumerate(run_data["smiles"]):
                score = self.score_molecule(smiles, query_name)
                props = self.calculate_molecular_properties(smiles)

                molecule_result = {
                    'SMILES': smiles,
                    'Oracle_Score': score,
                    'Query': query_name,
                    'Molecule_ID': f"{query_name}_it_{run_data['run']}_{i + 1}",
                    **props
                }
                scored_molecules.append(molecule_result)

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
                    "max_score": max(scores) if scores else 0,
                    "mean_score": np.mean(scores) if scores else 0,
                    "molecules": scored_molecules
                }

                evaluation_results["iterative"]["runs"].append(run_result)
                evaluation_results["iterative"]["auc_scores"].append(auc_top_10)
                evaluation_results["iterative"]["top_10_scores"].append(top_10_mean)

                print(f"  Run {run_data['run']}: {len(scored_molecules)} molecules, "
                      f"AUC-10: {auc_top_10:.4f}, Top-10: {top_10_mean:.4f}, "
                      f"Max: {max(scores):.4f}")

        return evaluation_results

    def create_publication_quality_plots(self, all_evaluations):
        """Create publication-quality visualizations focused on AUC Top-10"""
        print("\n📊 Creating publication-quality visualizations...")

        # Prepare data
        plot_data = []
        auc_data = []
        top10_progression_data = []

        for query_name, eval_data in all_evaluations.items():
            if eval_data is None:
                continue

            for pipeline in ['single_shot', 'iterative']:
                pipeline_name = pipeline.replace('_', '-').title()

                for run in eval_data[pipeline]['runs']:
                    # Individual molecule data
                    for mol in run['molecules']:
                        plot_data.append({
                            'Query': query_name,
                            'Pipeline': pipeline_name,
                            'Run': run['run'],
                            'Oracle_Score': mol['Oracle_Score'],
                            'MW': mol.get('MW', 0),
                            'LogP': mol.get('LogP', 0),
                            'TPSA': mol.get('TPSA', 0),
                            'QED': mol.get('QED', 0)
                        })

                    # AUC data
                    auc_data.append({
                        'Query': query_name,
                        'Pipeline': pipeline_name,
                        'Run': run['run'],
                        'AUC_Top10': run['auc_top_10'],
                        'Top10_Mean': run['top_10_mean'],
                        'Max_Score': run['max_score'],
                        'N_Molecules': len(run['molecules'])
                    })

                    # Top-10 progression data
                    top_scores = sorted([mol['Oracle_Score'] for mol in run['molecules']], reverse=True)[:10]
                    for rank, score in enumerate(top_scores, 1):
                        top10_progression_data.append({
                            'Query': query_name,
                            'Pipeline': pipeline_name,
                            'Run': run['run'],
                            'Rank': rank,
                            'Score': score
                        })

        if not plot_data:
            print("❌ No data for visualization")
            return None, None, None

        df = pd.DataFrame(plot_data)
        auc_df = pd.DataFrame(auc_data)
        top10_df = pd.DataFrame(top10_progression_data)

        # Create main figure
        fig = plt.figure(figsize=(24, 20))

        # 1. Main AUC Top-10 Results (Primary Plot)
        ax1 = plt.subplot(4, 4, (1, 2))  # Span 2 columns

        # Prepare AUC summary data
        auc_summary = auc_df.groupby(['Query', 'Pipeline']).agg({
            'AUC_Top10': ['mean', 'std', 'count']
        }).round(4)

        auc_summary.columns = ['AUC_Mean', 'AUC_Std', 'N_Runs']
        auc_summary = auc_summary.reset_index()

        # Create grouped bar plot
        sns.barplot(data=auc_summary, x='Query', y='AUC_Mean', hue='Pipeline', ax=ax1)
        ax1.set_title("AUC Top-10 Performance by Query and Pipeline", fontsize=16, fontweight='bold')
        ax1.set_ylabel("AUC Top-10 Score", fontsize=12)
        ax1.set_xlabel("Query Task", fontsize=12)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Add error bars
        x_offset = [-0.2, 0.2]  # Offset for grouped bars
        for i, (pipeline, group) in enumerate(auc_summary.groupby('Pipeline')):
            for j, (_, row) in enumerate(group.iterrows()):
                x_pos = j + x_offset[i]
                ax1.errorbar(x_pos, row['AUC_Mean'], yerr=row['AUC_Std'],
                             color='black', capsize=4, alpha=0.8, linewidth=1.5)

        ax1.legend(title='Pipeline', fontsize=10, title_fontsize=11)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Top-10 Score Progression
        ax2 = plt.subplot(4, 4, (3, 4))  # Span 2 columns

        # Show progression for top 5 performing queries
        top_queries = auc_df.groupby('Query')['AUC_Top10'].mean().nlargest(5).index

        colors = plt.cm.Set1(np.linspace(0, 1, len(top_queries)))

        for i, query in enumerate(top_queries):
            query_data = top10_df[top10_df['Query'] == query]

            for pipeline in query_data['Pipeline'].unique():
                pipeline_data = query_data[query_data['Pipeline'] == pipeline]
                avg_scores = pipeline_data.groupby('Rank')['Score'].mean()

                linestyle = '-' if pipeline == 'Single-Shot' else '--'
                ax2.plot(avg_scores.index, avg_scores.values,
                         color=colors[i], linestyle=linestyle, marker='o',
                         label=f"{query[:12]}_{pipeline}", alpha=0.8, linewidth=2)

        ax2.set_xlabel("Rank in Top-10", fontsize=12)
        ax2.set_ylabel("Oracle Score", fontsize=12)
        ax2.set_title("Top-10 Score Progression (Best 5 Queries)", fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(alpha=0.3)

        # 3. Pipeline Comparison Violin Plot
        ax3 = plt.subplot(4, 4, 5)
        sns.violinplot(data=auc_df, x='Pipeline', y='AUC_Top10', ax=ax3)
        ax3.set_title("AUC Distribution by Pipeline", fontsize=12, fontweight='bold')
        ax3.set_ylabel("AUC Top-10")

        # Add statistical annotation
        single_shot_auc = auc_df[auc_df['Pipeline'] == 'Single-Shot']['AUC_Top10']
        iterative_auc = auc_df[auc_df['Pipeline'] == 'Iterative']['AUC_Top10']

        if len(single_shot_auc) > 0 and len(iterative_auc) > 0:
            from scipy import stats
            stat, p_value = stats.mannwhitneyu(single_shot_auc, iterative_auc, alternative='two-sided')
            ax3.text(0.5, 0.95, f'p-value: {p_value:.4f}', transform=ax3.transAxes,
                     ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        # 4. Success Rate Analysis
        ax4 = plt.subplot(4, 4, 6)
        success_data = []
        for (query, pipeline), group in df.groupby(['Query', 'Pipeline']):
            high_scores = len(group[group['Oracle_Score'] > 0.8])
            total = len(group)
            success_rate = high_scores / total if total > 0 else 0
            success_data.append({
                'Query': query,
                'Pipeline': pipeline,
                'Success_Rate': success_rate * 100  # Convert to percentage
            })

        success_df = pd.DataFrame(success_data)
        sns.barplot(data=success_df, x='Query', y='Success_Rate', hue='Pipeline', ax=ax4)
        ax4.set_title("Success Rate (Score > 0.8)", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Success Rate (%)")
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        # 5. Best Scores by Query
        ax5 = plt.subplot(4, 4, 7)
        best_scores = df.groupby('Query')['Oracle_Score'].max().sort_values(ascending=False)
        best_scores.plot(kind='bar', ax=ax5, color='skyblue', alpha=0.8)
        ax5.set_title("Best Oracle Score by Query", fontsize=12, fontweight='bold')
        ax5.set_ylabel("Best Oracle Score")
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

        # 6. Molecular Properties Analysis
        ax6 = plt.subplot(4, 4, 8)
        scatter = ax6.scatter(df['MW'], df['Oracle_Score'],
                              c=df['LogP'], cmap='viridis', alpha=0.6, s=15)
        ax6.set_xlabel("Molecular Weight (Da)")
        ax6.set_ylabel("Oracle Score")
        ax6.set_title("MW vs Oracle Score (colored by LogP)")
        plt.colorbar(scatter, ax=ax6, label='LogP')

        # 7-10. Individual Query Detailed Analysis (Top 4 queries)
        top_4_queries = auc_df.groupby('Query')['AUC_Top10'].mean().nlargest(4).index

        for i, query in enumerate(top_4_queries):
            ax = plt.subplot(4, 4, 9 + i)
            query_auc = auc_df[auc_df['Query'] == query]

            # Box plot with individual points
            sns.boxplot(data=query_auc, x='Pipeline', y='AUC_Top10', ax=ax)
            sns.stripplot(data=query_auc, x='Pipeline', y='AUC_Top10',
                          ax=ax, color='red', alpha=0.7, size=4)

            ax.set_title(f"{query}\nAUC Top-10 Distribution", fontsize=10, fontweight='bold')
            ax.set_ylabel("AUC Top-10")

            # Add mean values as text
            for j, pipeline in enumerate(query_auc['Pipeline'].unique()):
                pipeline_data = query_auc[query_auc['Pipeline'] == pipeline]
                mean_val = pipeline_data['AUC_Top10'].mean()
                ax.text(j, ax.get_ylim()[1] * 0.9, f'μ={mean_val:.3f}',
                        ha='center', fontsize=8, fontweight='bold')

        # 11. Pipeline Efficiency Analysis
        ax11 = plt.subplot(4, 4, 13)
        efficiency_data = auc_df.groupby('Pipeline').agg({
            'AUC_Top10': 'mean',
            'N_Molecules': 'mean'
        }).reset_index()

        scatter = ax11.scatter(efficiency_data['N_Molecules'], efficiency_data['AUC_Top10'],
                               s=300, alpha=0.7, c=['blue', 'red'])

        for i, row in efficiency_data.iterrows():
            ax11.annotate(row['Pipeline'],
                          (row['N_Molecules'], row['AUC_Top10']),
                          xytext=(10, 10), textcoords='offset points',
                          fontsize=12, fontweight='bold')

        ax11.set_xlabel("Average Molecules Generated")
        ax11.set_ylabel("Average AUC Top-10")
        ax11.set_title("Pipeline Efficiency Analysis", fontsize=12, fontweight='bold')
        ax11.grid(alpha=0.3)

        # 12. Score Distribution Histogram
        ax12 = plt.subplot(4, 4, 14)
        for pipeline in df['Pipeline'].unique():
            subset = df[df['Pipeline'] == pipeline]
            ax12.hist(subset['Oracle_Score'], alpha=0.6, label=pipeline, bins=30, density=True)

        ax12.set_xlabel("Oracle Score")
        ax12.set_ylabel("Density")
        ax12.set_title("Oracle Score Distribution", fontsize=12, fontweight='bold')
        ax12.legend()
        ax12.grid(alpha=0.3)

        # 13. Query Difficulty vs Performance
        ax13 = plt.subplot(4, 4, 15)
        difficulty_data = df.groupby('Query').agg({
            'Oracle_Score': ['mean', 'max', 'std']
        }).round(3)

        difficulty_data.columns = ['Mean_Score', 'Max_Score', 'Score_Std']
        difficulty_data = difficulty_data.reset_index()

        # Merge with AUC data
        query_auc_means = auc_df.groupby('Query')['AUC_Top10'].mean()
        difficulty_data['AUC_Mean'] = difficulty_data['Query'].map(query_auc_means)

        scatter = ax13.scatter(difficulty_data['Score_Std'], difficulty_data['AUC_Mean'],
                               s=100, alpha=0.7, c=difficulty_data['Max_Score'], cmap='viridis')

        for i, row in difficulty_data.iterrows():
            if not pd.isna(row['AUC_Mean']):
                ax13.annotate(row['Query'][:8],
                              (row['Score_Std'], row['AUC_Mean']),
                              xytext=(2, 2), textcoords='offset points', fontsize=8)

        ax13.set_xlabel("Score Standard Deviation")
        ax13.set_ylabel("Average AUC Top-10")
        ax13.set_title("Query Difficulty Analysis", fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax13, label='Max Score')

        # 14. Run Consistency Analysis
        ax14 = plt.subplot(4, 4, 16)
        consistency_data = auc_df.groupby(['Query', 'Pipeline']).agg({
            'AUC_Top10': ['mean', 'std', 'count']
        }).round(4)

        consistency_data.columns = ['AUC_Mean', 'AUC_Std', 'N_Runs']
        consistency_data = consistency_data.reset_index()

        for pipeline in consistency_data['Pipeline'].unique():
            subset = consistency_data[consistency_data['Pipeline'] == pipeline]
            ax14.scatter(subset['AUC_Std'], subset['AUC_Mean'],
                         label=pipeline, alpha=0.7, s=80)

        ax14.set_xlabel("AUC Top-10 Standard Deviation")
        ax14.set_ylabel("AUC Top-10 Mean")
        ax14.set_title("Run Consistency Analysis", fontsize=12, fontweight='bold')
        ax14.legend()
        ax14.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / "comprehensive_oracle_evaluation.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

        return df, auc_df, top10_df

    def create_summary_tables(self, all_evaluations):
        """Create comprehensive summary tables"""
        print("\n📋 Creating Summary Tables...")

        # Prepare main results table
        main_results = []

        for query_name, eval_data in all_evaluations.items():
            if eval_data is None:
                continue

            oracle_name = eval_data["oracle_name"]

            # Single-shot results
            if eval_data["single_shot"]["auc_scores"]:
                ss_auc_scores = eval_data["single_shot"]["auc_scores"]
                ss_top10_scores = eval_data["single_shot"]["top_10_scores"]
                ss_max_scores = [run["max_score"] for run in eval_data["single_shot"]["runs"]]

                main_results.append({
                    'Query': query_name,
                    'Oracle': oracle_name,
                    'Pipeline': 'Single-Shot',
                    'AUC_Top10_Mean': np.mean(ss_auc_scores),
                    'AUC_Top10_Std': np.std(ss_auc_scores),
                    'Top10_Mean': np.mean(ss_top10_scores),
                    'Top10_Std': np.std(ss_top10_scores),
                    'Max_Score_Mean': np.mean(ss_max_scores),
                    'Max_Score_Max': np.max(ss_max_scores),
                    'N_Runs': len(ss_auc_scores),
                    'Total_Molecules': sum(len(run["molecules"]) for run in eval_data["single_shot"]["runs"])
                })

            # Iterative results
            if eval_data["iterative"]["auc_scores"]:
                it_auc_scores = eval_data["iterative"]["auc_scores"]
                it_top10_scores = eval_data["iterative"]["top_10_scores"]
                it_max_scores = [run["max_score"] for run in eval_data["iterative"]["runs"]]

                main_results.append({
                    'Query': query_name,
                    'Oracle': oracle_name,
                    'Pipeline': 'Iterative',
                    'AUC_Top10_Mean': np.mean(it_auc_scores),
                    'AUC_Top10_Std': np.std(it_auc_scores),
                    'Top10_Mean': np.mean(it_top10_scores),
                    'Top10_Std': np.std(it_top10_scores),
                    'Max_Score_Mean': np.mean(it_max_scores),
                    'Max_Score_Max': np.max(it_max_scores),
                    'N_Runs': len(it_auc_scores),
                    'Total_Molecules': sum(len(run["molecules"]) for run in eval_data["iterative"]["runs"])
                })

        main_df = pd.DataFrame(main_results)

        if len(main_df) > 0:
            # Save to CSV
            main_df.to_csv(self.results_dir / "comprehensive_oracle_results.csv", index=False)

            # Print formatted table
            print("\n" + "=" * 140)
            print("🏆 COMPREHENSIVE ORACLE EVALUATION RESULTS")
            print("=" * 140)
            print(
                f"{'Query':<25} {'Oracle':<20} {'Pipeline':<12} {'AUC-10':<10} {'±Std':<8} {'Top-10':<10} {'±Std':<8} {'Max':<8} {'Runs':<6} {'Mols':<6}")
            print("-" * 140)

            for _, row in main_df.iterrows():
                print(f"{row['Query']:<25} {row['Oracle']:<20} {row['Pipeline']:<12} "
                      f"{row['AUC_Top10_Mean']:<10.4f} ±{row['AUC_Top10_Std']:<7.4f} "
                      f"{row['Top10_Mean']:<10.4f} ±{row['Top10_Std']:<7.4f} "
                      f"{row['Max_Score_Max']:<8.4f} {row['N_Runs']:<6} {row['Total_Molecules']:<6}")

            # Statistical summaries
            print(f"\n🥇 BEST PERFORMANCE BY QUERY (AUC Top-10):")
            best_by_query = main_df.loc[main_df.groupby('Query')['AUC_Top10_Mean'].idxmax()]
            for _, row in best_by_query.iterrows():
                print(f"  {row['Query']:<25}: {row['Pipeline']:<12} "
                      f"AUC: {row['AUC_Top10_Mean']:.4f}±{row['AUC_Top10_Std']:.4f}")

            print(f"\n📊 PIPELINE PERFORMANCE SUMMARY:")
            pipeline_summary = main_df.groupby('Pipeline').agg({
                'AUC_Top10_Mean': ['mean', 'std', 'count'],
                'Top10_Mean': ['mean', 'std'],
                'Max_Score_Max': 'max',
                'Total_Molecules': 'sum'
            }).round(4)

            print(pipeline_summary)

            # Top performing tasks
            print(f"\n🎯 TOP 10 PERFORMING TASKS (by AUC Top-10):")
            top_10_tasks = main_df.nlargest(10, 'AUC_Top10_Mean')
            for i, (_, row) in enumerate(top_10_tasks.iterrows(), 1):
                print(f"{i:2d}. {row['Query']:<25} {row['Pipeline']:<12} "
                      f"AUC: {row['AUC_Top10_Mean']:.4f} (Oracle: {row['Oracle']})")

            print(f"\n💾 Results saved to: {self.results_dir / 'comprehensive_oracle_results.csv'}")

        return main_df

    def run_complete_evaluation(self):
        """Run the complete oracle evaluation pipeline"""
        print("🚀 Starting Complete Oracle Evaluation with TDC")
        print("=" * 80)

        # Extract experiment results
        experiment_results = self.extract_experiment_results()

        if not experiment_results:
            print("❌ No experiment results found!")
            return None

        print(f"✅ Found results for {len(experiment_results)} queries")

        # Evaluate each query
        all_evaluations = {}
        successful_evaluations = 0

        for query_name, query_results in experiment_results.items():
            print(f"\n🔍 Processing query: {query_name}")
            evaluation = self.evaluate_query_results(query_name, query_results)
            all_evaluations[query_name] = evaluation

            if evaluation is not None:
                successful_evaluations += 1

        print(f"\n✅ Successfully evaluated {successful_evaluations}/{len(experiment_results)} queries")

        # Create visualizations
        df, auc_df, top10_df = self.create_publication_quality_plots(all_evaluations)

        # Create summary tables
        summary_df = self.create_summary_tables(all_evaluations)

        # Save comprehensive results
        results_file = self.results_dir / "complete_tdc_oracle_evaluation.json"
        with open(results_file, 'w') as f:
            json.dump(all_evaluations, f, indent=2, default=str)

        print(f"\n🎉 Complete evaluation finished!")
        print(f"📊 Evaluated {successful_evaluations} queries with TDC oracles")
        print(f"📈 Generated comprehensive analysis and visualizations")
        print(f"💾 Detailed results saved to: {results_file}")

        return all_evaluations, df, auc_df, top10_df, summary_df

    def create_organized_visualizations(self, all_evaluations):
        """Create well-organized visualizations saved in separate folders"""
        print("\n📊 Creating organized visualizations...")

        # Create visualization directories
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (viz_dir / "main_results").mkdir(exist_ok=True)
        (viz_dir / "pipeline_comparison").mkdir(exist_ok=True)
        (viz_dir / "query_analysis").mkdir(exist_ok=True)
        (viz_dir / "molecular_properties").mkdir(exist_ok=True)
        (viz_dir / "detailed_analysis").mkdir(exist_ok=True)

        # Prepare data
        plot_data = []
        auc_data = []
        top10_progression_data = []

        for query_name, eval_data in all_evaluations.items():
            if eval_data is None:
                continue

            for pipeline in ['single_shot', 'iterative']:
                pipeline_name = pipeline.replace('_', '-').title()

                for run in eval_data[pipeline]['runs']:
                    # Individual molecule data
                    for mol in run['molecules']:
                        plot_data.append({
                            'Query': query_name,
                            'Pipeline': pipeline_name,
                            'Run': run['run'],
                            'Oracle_Score': mol['Oracle_Score'],
                            'MW': mol.get('MW', 0),
                            'LogP': mol.get('LogP', 0),
                            'TPSA': mol.get('TPSA', 0),
                            'QED': mol.get('QED', 0),
                            'HBD': mol.get('HBD', 0),
                            'HBA': mol.get('HBA', 0),
                            'RotBonds': mol.get('RotBonds', 0)
                        })

                    # AUC data
                    auc_data.append({
                        'Query': query_name,
                        'Pipeline': pipeline_name,
                        'Run': run['run'],
                        'AUC_Top10': run['auc_top_10'],
                        'Top10_Mean': run['top_10_mean'],
                        'Max_Score': run['max_score'],
                        'N_Molecules': len(run['molecules'])
                    })

                    # Top-10 progression data
                    top_scores = sorted([mol['Oracle_Score'] for mol in run['molecules']], reverse=True)[:10]
                    for rank, score in enumerate(top_scores, 1):
                        top10_progression_data.append({
                            'Query': query_name,
                            'Pipeline': pipeline_name,
                            'Run': run['run'],
                            'Rank': rank,
                            'Score': score
                        })

        if not plot_data:
            print("❌ No data for visualization")
            return None, None, None

        df = pd.DataFrame(plot_data)
        auc_df = pd.DataFrame(auc_data)
        top10_df = pd.DataFrame(top10_progression_data)

        # 1. MAIN RESULTS VISUALIZATIONS
        self._create_main_results_plots(auc_df, viz_dir / "main_results")

        # 2. PIPELINE COMPARISON VISUALIZATIONS
        self._create_pipeline_comparison_plots(df, auc_df, viz_dir / "pipeline_comparison")

        # 3. QUERY ANALYSIS VISUALIZATIONS
        self._create_query_analysis_plots(df, auc_df, top10_df, viz_dir / "query_analysis")

        # 4. MOLECULAR PROPERTIES VISUALIZATIONS
        self._create_molecular_properties_plots(df, viz_dir / "molecular_properties")

        # 5. DETAILED ANALYSIS VISUALIZATIONS
        self._create_detailed_analysis_plots(df, auc_df, viz_dir / "detailed_analysis")

        return df, auc_df, top10_df

    def _create_main_results_plots(self, auc_df, save_dir):
        """Create main results visualizations"""
        print("📈 Creating main results plots...")

        # 1. Primary AUC Top-10 Results
        plt.figure(figsize=(16, 8))

        # Prepare AUC summary data
        auc_summary = auc_df.groupby(['Query', 'Pipeline']).agg({
            'AUC_Top10': ['mean', 'std', 'count']
        }).round(4)

        auc_summary.columns = ['AUC_Mean', 'AUC_Std', 'N_Runs']
        auc_summary = auc_summary.reset_index()

        # Create grouped bar plot
        ax = sns.barplot(data=auc_summary, x='Query', y='AUC_Mean', hue='Pipeline')
        ax.set_title("AUC Top-10 Performance by Query and Pipeline", fontsize=16, fontweight='bold')
        ax.set_ylabel("AUC Top-10 Score", fontsize=12)
        ax.set_xlabel("Query Task", fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Add error bars
        x_offset = [-0.2, 0.2]
        for i, (pipeline, group) in enumerate(auc_summary.groupby('Pipeline')):
            for j, (_, row) in enumerate(group.iterrows()):
                x_pos = j + x_offset[i]
                ax.errorbar(x_pos, row['AUC_Mean'], yerr=row['AUC_Std'],
                            color='black', capsize=4, alpha=0.8, linewidth=1.5)

        plt.legend(title='Pipeline', fontsize=10, title_fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "01_main_auc_results.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Top Performers Summary
        plt.figure(figsize=(12, 8))
        top_10_tasks = auc_summary.nlargest(10, 'AUC_Mean')

        colors = ['#1f77b4' if x == 'Single-Shot' else '#ff7f0e' for x in top_10_tasks['Pipeline']]
        bars = plt.bar(range(len(top_10_tasks)), top_10_tasks['AUC_Mean'], color=colors, alpha=0.8)

        plt.title("Top 10 Performing Tasks (AUC Top-10)", fontsize=16, fontweight='bold')
        plt.ylabel("AUC Top-10 Score", fontsize=12)
        plt.xlabel("Task Rank", fontsize=12)

        # Add value labels on bars
        for i, (bar, row) in enumerate(zip(bars, top_10_tasks.iterrows())):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{row[1]["AUC_Mean"]:.3f}', ha='center', va='bottom', fontsize=10)

        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Single-Shot'),
                           Patch(facecolor='#ff7f0e', label='Iterative')]
        plt.legend(handles=legend_elements, title='Pipeline')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "02_top_performers.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_pipeline_comparison_plots(self, df, auc_df, save_dir):
        """Create pipeline comparison visualizations"""
        print("🔄 Creating pipeline comparison plots...")

        # 1. Pipeline Distribution Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # AUC Distribution
        sns.violinplot(data=auc_df, x='Pipeline', y='AUC_Top10', ax=axes[0, 0])
        axes[0, 0].set_title("AUC Top-10 Distribution by Pipeline", fontweight='bold')

        # Statistical test
        if len(auc_df['Pipeline'].unique()) == 2:
            from scipy import stats
            single_shot = auc_df[auc_df['Pipeline'] == 'Single-Shot']['AUC_Top10']
            iterative = auc_df[auc_df['Pipeline'] == 'Iterative']['AUC_Top10']
            if len(single_shot) > 0 and len(iterative) > 0:
                stat, p_value = stats.mannwhitneyu(single_shot, iterative, alternative='two-sided')
                axes[0, 0].text(0.5, 0.95, f'p-value: {p_value:.4f}',
                                transform=axes[0, 0].transAxes, ha='center', fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        # Oracle Score Distribution
        sns.violinplot(data=df, x='Pipeline', y='Oracle_Score', ax=axes[0, 1])
        axes[0, 1].set_title("Oracle Score Distribution by Pipeline", fontweight='bold')

        # Success Rate Analysis
        success_data = []
        for (query, pipeline), group in df.groupby(['Query', 'Pipeline']):
            high_scores = len(group[group['Oracle_Score'] > 0.8])
            total = len(group)
            success_rate = high_scores / total if total > 0 else 0
            success_data.append({
                'Query': query,
                'Pipeline': pipeline,
                'Success_Rate': success_rate * 100
            })

        success_df = pd.DataFrame(success_data)
        sns.boxplot(data=success_df, x='Pipeline', y='Success_Rate', ax=axes[1, 0])
        axes[1, 0].set_title("Success Rate Distribution (Score > 0.8)", fontweight='bold')
        axes[1, 0].set_ylabel("Success Rate (%)")

        # Efficiency Analysis
        efficiency_data = auc_df.groupby('Pipeline').agg({
            'AUC_Top10': 'mean',
            'N_Molecules': 'mean'
        }).reset_index()

        colors = ['#1f77b4', '#ff7f0e']
        for i, (_, row) in enumerate(efficiency_data.iterrows()):
            axes[1, 1].scatter(row['N_Molecules'], row['AUC_Top10'],
                               s=300, alpha=0.7, c=colors[i], label=row['Pipeline'])
            axes[1, 1].annotate(row['Pipeline'],
                                (row['N_Molecules'], row['AUC_Top10']),
                                xytext=(10, 10), textcoords='offset points',
                                fontsize=12, fontweight='bold')

        axes[1, 1].set_xlabel("Average Molecules Generated")
        axes[1, 1].set_ylabel("Average AUC Top-10")
        axes[1, 1].set_title("Pipeline Efficiency Analysis", fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "01_pipeline_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_query_analysis_plots(self, df, auc_df, top10_df, save_dir):
        """Create query-specific analysis plots"""
        print("🎯 Creating query analysis plots...")

        # 1. Query Performance Heatmap
        plt.figure(figsize=(14, 8))

        # Create pivot table for heatmap
        heatmap_data = auc_df.pivot_table(values='AUC_Top10',
                                          index='Query',
                                          columns='Pipeline',
                                          aggfunc='mean')

        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                    cbar_kws={'label': 'AUC Top-10'})
        plt.title("Query Performance Heatmap (AUC Top-10)", fontsize=16, fontweight='bold')
        plt.ylabel("Query Task", fontsize=12)
        plt.xlabel("Pipeline", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / "01_query_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Top-10 Score Progression for Best Queries
        plt.figure(figsize=(12, 8))

        top_queries = auc_df.groupby('Query')['AUC_Top10'].mean().nlargest(8).index
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_queries)))

        for i, query in enumerate(top_queries):
            query_data = top10_df[top10_df['Query'] == query]

            for pipeline in query_data['Pipeline'].unique():
                pipeline_data = query_data[query_data['Pipeline'] == pipeline]
                avg_scores = pipeline_data.groupby('Rank')['Score'].mean()

                linestyle = '-' if pipeline == 'Single-Shot' else '--'
                plt.plot(avg_scores.index, avg_scores.values,
                         color=colors[i], linestyle=linestyle, marker='o',
                         label=f"{query[:15]}_{pipeline}", alpha=0.8, linewidth=2)

        plt.xlabel("Rank in Top-10", fontsize=12)
        plt.ylabel("Oracle Score", fontsize=12)
        plt.title("Top-10 Score Progression (Best 8 Queries)", fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "02_top10_progression.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_molecular_properties_plots(self, df, save_dir):
        """Create molecular properties analysis plots"""
        print("🧪 Creating molecular properties plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. MW vs Oracle Score
        scatter = axes[0, 0].scatter(df['MW'], df['Oracle_Score'],
                                     c=df['LogP'], cmap='viridis', alpha=0.6, s=15)
        axes[0, 0].set_xlabel("Molecular Weight (Da)")
        axes[0, 0].set_ylabel("Oracle Score")
        axes[0, 0].set_title("MW vs Oracle Score (colored by LogP)")
        plt.colorbar(scatter, ax=axes[0, 0], label='LogP')

        # 2. LogP vs Oracle Score
        scatter2 = axes[0, 1].scatter(df['LogP'], df['Oracle_Score'],
                                      c=df['TPSA'], cmap='plasma', alpha=0.6, s=15)
        axes[0, 1].set_xlabel("LogP")
        axes[0, 1].set_ylabel("Oracle Score")
        axes[0, 1].set_title("LogP vs Oracle Score (colored by TPSA)")
        plt.colorbar(scatter2, ax=axes[0, 1], label='TPSA')

        # 3. TPSA vs Oracle Score
        scatter3 = axes[0, 2].scatter(df['TPSA'], df['Oracle_Score'],
                                      c=df['QED'], cmap='coolwarm', alpha=0.6, s=15)
        axes[0, 2].set_xlabel("TPSA")
        axes[0, 2].set_ylabel("Oracle Score")
        axes[0, 2].set_title("TPSA vs Oracle Score (colored by QED)")
        plt.colorbar(scatter3, ax=axes[0, 2], label='QED')

        # 4. Property distributions by pipeline
        sns.boxplot(data=df, x='Pipeline', y='MW', ax=axes[1, 0])
        axes[1, 0].set_title("Molecular Weight Distribution")

        sns.boxplot(data=df, x='Pipeline', y='LogP', ax=axes[1, 1])
        axes[1, 1].set_title("LogP Distribution")

        sns.boxplot(data=df, x='Pipeline', y='QED', ax=axes[1, 2])
        axes[1, 2].set_title("QED Distribution")

        plt.tight_layout()
        plt.savefig(save_dir / "01_molecular_properties.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_detailed_analysis_plots(self, df, auc_df, save_dir):
        """Create detailed analysis plots"""
        print("🔍 Creating detailed analysis plots...")

        # 1. Score Distribution Histogram
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        for pipeline in df['Pipeline'].unique():
            subset = df[df['Pipeline'] == pipeline]
            plt.hist(subset['Oracle_Score'], alpha=0.6, label=pipeline, bins=30, density=True)

        plt.xlabel("Oracle Score")
        plt.ylabel("Density")
        plt.title("Oracle Score Distribution by Pipeline")
        plt.legend()
        plt.grid(alpha=0.3)

        # 2. Run Consistency Analysis
        plt.subplot(1, 2, 2)
        consistency_data = auc_df.groupby(['Query', 'Pipeline']).agg({
            'AUC_Top10': ['mean', 'std', 'count']
        }).round(4)

        consistency_data.columns = ['AUC_Mean', 'AUC_Std', 'N_Runs']
        consistency_data = consistency_data.reset_index()

        for pipeline in consistency_data['Pipeline'].unique():
            subset = consistency_data[consistency_data['Pipeline'] == pipeline]
            plt.scatter(subset['AUC_Std'], subset['AUC_Mean'],
                        label=pipeline, alpha=0.7, s=80)

        plt.xlabel("AUC Top-10 Standard Deviation")
        plt.ylabel("AUC Top-10 Mean")
        plt.title("Run Consistency Analysis")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "01_detailed_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_comprehensive_result_tables(self, all_evaluations):
        """Create comprehensive result tables with ALL 23 tasks included"""
        print("\n📋 Creating Comprehensive Result Tables...")

        # Create tables directory
        tables_dir = self.results_dir / "result_tables"
        tables_dir.mkdir(exist_ok=True)

        # Get all possible tasks from the oracle mapping
        all_possible_tasks = set(COMPLETE_ORACLE_MAPPING.keys())

        print(f"🔍 Processing all {len(all_possible_tasks)} tasks...")

        # Prepare data for tables - INCLUDE ALL TASKS
        single_shot_results = []
        iterative_results = []

        for task_name in sorted(all_possible_tasks):  # Process ALL 23 tasks in sorted order
            eval_data = all_evaluations.get(task_name)
            oracle_name = COMPLETE_ORACLE_MAPPING[task_name]

            print(f"   Processing: {task_name}")

            if eval_data is None:
                print(f"      ❌ No evaluation data - adding zeros")
                # Task failed evaluation - add with zeros for BOTH pipelines
                single_shot_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': 0.0,
                    'AUC_Top10_Std': 0.0,
                    'AUC_Top10_Min': 0.0,
                    'AUC_Top10_Max': 0.0,
                    'Top10_Mean': 0.0,
                    'Top10_Std': 0.0,
                    'Best_Score': 0.0,
                    'Avg_Max_Score': 0.0,
                    'N_Runs': 0,
                    'Total_Molecules': 0,
                    'Avg_Molecules_Per_Run': 0.0
                })

                iterative_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': 0.0,
                    'AUC_Top10_Std': 0.0,
                    'AUC_Top10_Min': 0.0,
                    'AUC_Top10_Max': 0.0,
                    'Top10_Mean': 0.0,
                    'Top10_Std': 0.0,
                    'Best_Score': 0.0,
                    'Avg_Max_Score': 0.0,
                    'N_Runs': 0,
                    'Total_Molecules': 0,
                    'Avg_Molecules_Per_Run': 0.0
                })
                continue

            # Single-shot results
            if eval_data["single_shot"]["auc_scores"]:
                print(f"      ✅ Single-shot: {len(eval_data['single_shot']['auc_scores'])} runs")
                ss_auc_scores = eval_data["single_shot"]["auc_scores"]
                ss_top10_scores = eval_data["single_shot"]["top_10_scores"]
                ss_max_scores = [run["max_score"] for run in eval_data["single_shot"]["runs"]]
                ss_total_molecules = sum(len(run["molecules"]) for run in eval_data["single_shot"]["runs"])

                single_shot_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': np.mean(ss_auc_scores),
                    'AUC_Top10_Std': np.std(ss_auc_scores),
                    'AUC_Top10_Min': np.min(ss_auc_scores),
                    'AUC_Top10_Max': np.max(ss_auc_scores),
                    'Top10_Mean': np.mean(ss_top10_scores),
                    'Top10_Std': np.std(ss_top10_scores),
                    'Best_Score': np.max(ss_max_scores),
                    'Avg_Max_Score': np.mean(ss_max_scores),
                    'N_Runs': len(ss_auc_scores),
                    'Total_Molecules': ss_total_molecules,
                    'Avg_Molecules_Per_Run': ss_total_molecules / len(ss_auc_scores)
                })
            else:
                print(f"      ⚠️ Single-shot: 0 runs - adding zeros")
                # No single-shot results - add with zeros
                single_shot_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': 0.0,
                    'AUC_Top10_Std': 0.0,
                    'AUC_Top10_Min': 0.0,
                    'AUC_Top10_Max': 0.0,
                    'Top10_Mean': 0.0,
                    'Top10_Std': 0.0,
                    'Best_Score': 0.0,
                    'Avg_Max_Score': 0.0,
                    'N_Runs': 0,
                    'Total_Molecules': 0,
                    'Avg_Molecules_Per_Run': 0.0
                })

            # Iterative results
            if eval_data["iterative"]["auc_scores"]:
                print(f"      ✅ Iterative: {len(eval_data['iterative']['auc_scores'])} runs")
                it_auc_scores = eval_data["iterative"]["auc_scores"]
                it_top10_scores = eval_data["iterative"]["top_10_scores"]
                it_max_scores = [run["max_score"] for run in eval_data["iterative"]["runs"]]
                it_total_molecules = sum(len(run["molecules"]) for run in eval_data["iterative"]["runs"])

                iterative_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': np.mean(it_auc_scores),
                    'AUC_Top10_Std': np.std(it_auc_scores),
                    'AUC_Top10_Min': np.min(it_auc_scores),
                    'AUC_Top10_Max': np.max(it_auc_scores),
                    'Top10_Mean': np.mean(it_top10_scores),
                    'Top10_Std': np.std(it_top10_scores),
                    'Best_Score': np.max(it_max_scores),
                    'Avg_Max_Score': np.mean(it_max_scores),
                    'N_Runs': len(it_auc_scores),
                    'Total_Molecules': it_total_molecules,
                    'Avg_Molecules_Per_Run': it_total_molecules / len(it_auc_scores)
                })
            else:
                print(f"      ⚠️ Iterative: 0 runs - adding zeros")
                # No iterative results - add with zeros
                iterative_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': 0.0,
                    'AUC_Top10_Std': 0.0,
                    'AUC_Top10_Min': 0.0,
                    'AUC_Top10_Max': 0.0,
                    'Top10_Mean': 0.0,
                    'Top10_Std': 0.0,
                    'Best_Score': 0.0,
                    'Avg_Max_Score': 0.0,
                    'N_Runs': 0,
                    'Total_Molecules': 0,
                    'Avg_Molecules_Per_Run': 0.0
                })

        # Create DataFrames
        ss_df = pd.DataFrame(single_shot_results)
        it_df = pd.DataFrame(iterative_results)

        # Verify we have all 23 tasks
        print(f"\n🔍 VERIFICATION:")
        print(f"   Single-Shot DataFrame: {len(ss_df)} tasks")
        print(f"   Iterative DataFrame: {len(it_df)} tasks")
        print(f"   Expected: 23 tasks each")

        if len(ss_df) != 23 or len(it_df) != 23:
            print(f"❌ ERROR: Expected 23 tasks, got SS={len(ss_df)}, IT={len(it_df)}")

        # Sort by AUC Top-10 Mean (descending)
        ss_df = ss_df.sort_values('AUC_Top10_Mean', ascending=False)
        it_df = it_df.sort_values('AUC_Top10_Mean', ascending=False)

        # Manual verification
        print(f"   Single-Shot AUC Sum: {ss_df['AUC_Top10_Mean'].sum():.4f}")
        print(f"   Iterative AUC Sum: {it_df['AUC_Top10_Mean'].sum():.4f}")

        # Show zero-score tasks
        ss_zeros = ss_df[ss_df['AUC_Top10_Mean'] == 0.0]['Query'].tolist()
        it_zeros = it_df[it_df['AUC_Top10_Mean'] == 0.0]['Query'].tolist()

        print(f"   Single-Shot zero scores: {ss_zeros}")
        print(f"   Iterative zero scores: {it_zeros}")

        # Save to CSV
        ss_df.to_csv(tables_dir / "single_shot_ALL_23_tasks.csv", index=False)
        it_df.to_csv(tables_dir / "iterative_ALL_23_tasks.csv", index=False)

        # Create formatted tables
        self._create_formatted_tables(ss_df, it_df, tables_dir)

        return ss_df, it_df

    def _create_formatted_tables(self, ss_df, it_df, tables_dir):
        """Create beautifully formatted tables with ALL tasks, not just top 10"""

        # 1. Single-Shot ALL TASKS AUC Table
        print("\n" + "=" * 120)
        print("🥇 SINGLE-SHOT PIPELINE - ALL TASKS AUC RESULTS")
        print("=" * 120)
        print(
            f"{'Rank':<4} {'Query':<25} {'Oracle':<20} {'AUC-10':<8} {'±Std':<8} {'Top-10':<8} {'Best':<8} {'Runs':<5} {'Mols':<5}")
        print("-" * 120)

        # Show ALL tasks, not just top 10
        for i, (_, row) in enumerate(ss_df.iterrows(), 1):
            print(f"{i:<4} {row['Query']:<25} {row['Oracle']:<20} "
                  f"{row['AUC_Top10_Mean']:<8.4f} ±{row['AUC_Top10_Std']:<7.4f} "
                  f"{row['Top10_Mean']:<8.4f} {row['Best_Score']:<8.4f} "
                  f"{row['N_Runs']:<5} {row['Total_Molecules']:<5}")

        # CORRECTED SUMMARY CALCULATIONS
        ss_summary = {
            'AUC_Sum': ss_df['AUC_Top10_Mean'].sum(),  # SUM of all AUC scores
            'AUC_Mean': ss_df['AUC_Top10_Mean'].mean(),  # MEAN of all AUC scores
            'AUC_Std': ss_df['AUC_Top10_Mean'].std(),
            'Top10_Sum': ss_df['Top10_Mean'].sum(),  # SUM of all Top-10 scores
            'Top10_Mean': ss_df['Top10_Mean'].mean(),  # MEAN of all Top-10 scores
            'Best_Overall': ss_df['Best_Score'].max(),
            'Total_Runs': ss_df['N_Runs'].sum(),
            'Total_Molecules': ss_df['Total_Molecules'].sum(),
            'Total_Queries': len(ss_df)
        }

        print("-" * 120)
        # Fix the f-string syntax by separating the variables
        tasks_label = f"({ss_summary['Total_Queries']} TASKS)"
        print(f"{'SUM':<4} {'SINGLE-SHOT TOTALS':<25} {tasks_label:<20} "
              f"{ss_summary['AUC_Sum']:<8.4f} ±{ss_summary['AUC_Std']:<7.4f} "
              f"{ss_summary['Top10_Sum']:<8.4f} {ss_summary['Best_Overall']:<8.4f} "
              f"{ss_summary['Total_Runs']:<5} {ss_summary['Total_Molecules']:<5}")

        print(f"{'AVG':<4} {'SINGLE-SHOT AVERAGE':<25} {'PER TASK':<20} "
              f"{ss_summary['AUC_Mean']:<8.4f} ±{ss_summary['AUC_Std']:<7.4f} "
              f"{ss_summary['Top10_Mean']:<8.4f} {ss_summary['Best_Overall']:<8.4f} "
              f"{ss_summary['Total_Runs'] / ss_summary['Total_Queries']:<5.1f} {ss_summary['Total_Molecules'] / ss_summary['Total_Queries']:<5.1f}")

        # 2. Iterative ALL TASKS AUC Table
        print("\n" + "=" * 120)
        print("🚀 ITERATIVE PIPELINE - ALL TASKS AUC RESULTS")
        print("=" * 120)
        print(
            f"{'Rank':<4} {'Query':<25} {'Oracle':<20} {'AUC-10':<8} {'±Std':<8} {'Top-10':<8} {'Best':<8} {'Runs':<5} {'Mols':<5}")
        print("-" * 120)

        # Show ALL tasks, not just top 10
        for i, (_, row) in enumerate(it_df.iterrows(), 1):
            print(f"{i:<4} {row['Query']:<25} {row['Oracle']:<20} "
                  f"{row['AUC_Top10_Mean']:<8.4f} ±{row['AUC_Top10_Std']:<7.4f} "
                  f"{row['Top10_Mean']:<8.4f} {row['Best_Score']:<8.4f} "
                  f"{row['N_Runs']:<5} {row['Total_Molecules']:<5}")

        # CORRECTED SUMMARY CALCULATIONS
        it_summary = {
            'AUC_Sum': it_df['AUC_Top10_Mean'].sum(),  # SUM of all AUC scores
            'AUC_Mean': it_df['AUC_Top10_Mean'].mean(),  # MEAN of all AUC scores
            'AUC_Std': it_df['AUC_Top10_Mean'].std(),
            'Top10_Sum': it_df['Top10_Mean'].sum(),  # SUM of all Top-10 scores
            'Top10_Mean': it_df['Top10_Mean'].mean(),  # MEAN of all Top-10 scores
            'Best_Overall': it_df['Best_Score'].max(),
            'Total_Runs': it_df['N_Runs'].sum(),
            'Total_Molecules': it_df['Total_Molecules'].sum(),
            'Total_Queries': len(it_df)
        }

        print("-" * 120)
        # Fix the f-string syntax by separating the variables
        it_tasks_label = f"({it_summary['Total_Queries']} TASKS)"
        print(f"{'SUM':<4} {'ITERATIVE TOTALS':<25} {it_tasks_label:<20} "
              f"{it_summary['AUC_Sum']:<8.4f} ±{it_summary['AUC_Std']:<7.4f} "
              f"{it_summary['Top10_Sum']:<8.4f} {it_summary['Best_Overall']:<8.4f} "
              f"{it_summary['Total_Runs']:<5} {it_summary['Total_Molecules']:<5}")

        print(f"{'AVG':<4} {'ITERATIVE AVERAGE':<25} {'PER TASK':<20} "
              f"{it_summary['AUC_Mean']:<8.4f} ±{it_summary['AUC_Std']:<7.4f} "
              f"{it_summary['Top10_Mean']:<8.4f} {it_summary['Best_Overall']:<8.4f} "
              f"{it_summary['Total_Runs'] / it_summary['Total_Queries']:<5.1f} {it_summary['Total_Molecules'] / it_summary['Total_Queries']:<5.1f}")

        # 3. COMBINED SUMMARY TABLE FOR ALL TASKS
        print("\n" + "=" * 120)
        print("📊 COMBINED PIPELINE COMPARISON - ALL TASKS")
        print("=" * 120)
        print(
            f"{'Pipeline':<15} {'Tasks':<6} {'AUC_Sum':<10} {'AUC_Avg':<10} {'Top10_Sum':<10} {'Best':<8} {'Runs':<6} {'Mols':<6}")
        print("-" * 120)

        print(f"{'Single-Shot':<15} {ss_summary['Total_Queries']:<6} "
              f"{ss_summary['AUC_Sum']:<10.4f} {ss_summary['AUC_Mean']:<10.4f} "
              f"{ss_summary['Top10_Sum']:<10.4f} {ss_summary['Best_Overall']:<8.4f} "
              f"{ss_summary['Total_Runs']:<6} {ss_summary['Total_Molecules']:<6}")

        print(f"{'Iterative':<15} {it_summary['Total_Queries']:<6} "
              f"{it_summary['AUC_Sum']:<10.4f} {it_summary['AUC_Mean']:<10.4f} "
              f"{it_summary['Top10_Sum']:<10.4f} {it_summary['Best_Overall']:<8.4f} "
              f"{it_summary['Total_Runs']:<6} {it_summary['Total_Molecules']:<6}")

        # Calculate differences
        auc_diff = it_summary['AUC_Sum'] - ss_summary['AUC_Sum']
        auc_avg_diff = it_summary['AUC_Mean'] - ss_summary['AUC_Mean']
        top10_diff = it_summary['Top10_Sum'] - ss_summary['Top10_Sum']
        task_diff = it_summary['Total_Queries'] - ss_summary['Total_Queries']

        print("-" * 120)
        print(f"{'DIFFERENCE':<15} {task_diff:<+6} "
              f"{auc_diff:<+10.4f} {auc_avg_diff:<+10.4f} "
              f"{top10_diff:<+10.4f} {'=':<8} "
              f"{it_summary['Total_Runs'] - ss_summary['Total_Runs']:<+6} {it_summary['Total_Molecules'] - ss_summary['Total_Molecules']:<+6}")

        # Determine winners
        auc_sum_winner = "Iterative" if auc_diff > 0 else "Single-Shot" if auc_diff < 0 else "Tie"
        auc_avg_winner = "Iterative" if auc_avg_diff > 0 else "Single-Shot" if auc_avg_diff < 0 else "Tie"

        print(f"\n🏆 FINAL RESULTS:")
        print(f"   AUC Sum Winner: {auc_sum_winner} (Difference: {auc_diff:+.4f})")
        print(f"   AUC Avg Winner: {auc_avg_winner} (Difference: {auc_avg_diff:+.4f})")
        print(f"   Task Coverage: Single-Shot={ss_summary['Total_Queries']}, Iterative={it_summary['Total_Queries']}")

        # 4. Show tasks that are missing from either pipeline
        all_ss_tasks = set(ss_df['Query'].tolist())
        all_it_tasks = set(it_df['Query'].tolist())

        ss_only = all_ss_tasks - all_it_tasks
        it_only = all_it_tasks - all_ss_tasks
        common_tasks = all_ss_tasks & all_it_tasks

        print(f"\n📋 TASK COVERAGE ANALYSIS:")
        print(f"   Common Tasks (both pipelines): {len(common_tasks)}")
        print(f"   Single-Shot Only: {len(ss_only)} tasks")
        if ss_only:
            print(f"      {', '.join(sorted(ss_only))}")
        print(f"   Iterative Only: {len(it_only)} tasks")
        if it_only:
            print(f"      {', '.join(sorted(it_only))}")

        # Save complete results to files (ALL TASKS, not just top 10)
        with open(tables_dir / "single_shot_ALL_tasks.txt", 'w') as f:
            f.write("SINGLE-SHOT PIPELINE - ALL TASKS RESULTS\n")
            f.write("=" * 120 + "\n")
            f.write(
                f"{'Rank':<4} {'Query':<25} {'Oracle':<20} {'AUC-10':<8} {'±Std':<8} {'Top-10':<8} {'Best':<8} {'Runs':<5} {'Mols':<5}\n")
            f.write("-" * 120 + "\n")

            for i, (_, row) in enumerate(ss_df.iterrows(), 1):
                f.write(f"{i:<4} {row['Query']:<25} {row['Oracle']:<20} "
                        f"{row['AUC_Top10_Mean']:<8.4f} ±{row['AUC_Top10_Std']:<7.4f} "
                        f"{row['Top10_Mean']:<8.4f} {row['Best_Score']:<8.4f} "
                        f"{row['N_Runs']:<5} {row['Total_Molecules']:<5}\n")

            f.write("-" * 120 + "\n")
            ss_tasks_label = f"({ss_summary['Total_Queries']} TASKS)"
            f.write(f"{'SUM':<4} {'SINGLE-SHOT TOTALS':<25} {ss_tasks_label:<20} "
                    f"{ss_summary['AUC_Sum']:<8.4f} ±{ss_summary['AUC_Std']:<7.4f} "
                    f"{ss_summary['Top10_Sum']:<8.4f} {ss_summary['Best_Overall']:<8.4f} "
                    f"{ss_summary['Total_Runs']:<5} {ss_summary['Total_Molecules']:<5}\n")

        with open(tables_dir / "iterative_ALL_tasks.txt", 'w') as f:
            f.write("ITERATIVE PIPELINE - ALL TASKS RESULTS\n")
            f.write("=" * 120 + "\n")
            f.write(
                f"{'Rank':<4} {'Query':<25} {'Oracle':<20} {'AUC-10':<8} {'±Std':<8} {'Top-10':<8} {'Best':<8} {'Runs':<5} {'Mols':<5}\n")
            f.write("-" * 120 + "\n")

            for i, (_, row) in enumerate(it_df.iterrows(), 1):
                f.write(f"{i:<4} {row['Query']:<25} {row['Oracle']:<20} "
                        f"{row['AUC_Top10_Mean']:<8.4f} ±{row['AUC_Top10_Std']:<7.4f} "
                        f"{row['Top10_Mean']:<8.4f} {row['Best_Score']:<8.4f} "
                        f"{row['N_Runs']:<5} {row['Total_Molecules']:<5}\n")

            f.write("-" * 120 + "\n")
            it_tasks_label = f"({it_summary['Total_Queries']} TASKS)"
            f.write(f"{'SUM':<4} {'ITERATIVE TOTALS':<25} {it_tasks_label:<20} "
                    f"{it_summary['AUC_Sum']:<8.4f} ±{it_summary['AUC_Std']:<7.4f} "
                    f"{it_summary['Top10_Sum']:<8.4f} {it_summary['Best_Overall']:<8.4f} "
                    f"{it_summary['Total_Runs']:<5} {it_summary['Total_Molecules']:<5}\n")

        return ss_summary, it_summary

    def analyze_missing_tasks(self, all_evaluations):
        """Analyze which tasks are missing and why"""
        print("\n🔍 ANALYZING MISSING TASKS...")

        total_expected = 23  # You mentioned 23 tasks
        total_found = len([eval_data for eval_data in all_evaluations.values() if eval_data is not None])

        print(f"Expected Tasks: {total_expected}")
        print(f"Found Tasks: {total_found}")
        print(f"Missing Tasks: {total_expected - total_found}")

        # Show which tasks are missing
        all_task_names = set(all_evaluations.keys())
        expected_tasks = set(COMPLETE_ORACLE_MAPPING.keys())

        missing_from_evaluation = expected_tasks - all_task_names
        failed_evaluations = [name for name, eval_data in all_evaluations.items() if eval_data is None]

        if missing_from_evaluation:
            print(f"\n❌ Tasks missing from experiment results: {len(missing_from_evaluation)}")
            for task in sorted(missing_from_evaluation):
                print(f"   - {task}")

        if failed_evaluations:
            print(f"\n⚠️ Tasks that failed evaluation: {len(failed_evaluations)}")
            for task in sorted(failed_evaluations):
                print(f"   - {task} (Oracle loading failed)")

        # Show tasks with no single-shot or iterative results
        tasks_with_ss = []
        tasks_with_it = []

        for name, eval_data in all_evaluations.items():
            if eval_data is not None:
                if eval_data["single_shot"]["auc_scores"]:
                    tasks_with_ss.append(name)
                if eval_data["iterative"]["auc_scores"]:
                    tasks_with_it.append(name)

        print(f"\n📊 PIPELINE COVERAGE:")
        print(f"   Tasks with Single-Shot results: {len(tasks_with_ss)}")
        print(f"   Tasks with Iterative results: {len(tasks_with_it)}")

        ss_only = set(tasks_with_ss) - set(tasks_with_it)
        it_only = set(tasks_with_it) - set(tasks_with_ss)

        if ss_only:
            print(f"   Single-Shot only: {list(ss_only)}")
        if it_only:
            print(f"   Iterative only: {list(it_only)}")

    def _create_summary_statistics(self, ss_df, it_df, tables_dir):
        """Create comprehensive summary statistics"""

        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE SUMMARY STATISTICS")
        print("=" * 100)

        # Overall comparison
        print(f"\n🏆 OVERALL PERFORMANCE COMPARISON:")
        print(f"{'Metric':<30} {'Single-Shot':<15} {'Iterative':<15} {'Winner':<15}")
        print("-" * 75)

        metrics = [
            ('Average AUC Top-10', ss_df['AUC_Top10_Mean'].mean(), it_df['AUC_Top10_Mean'].mean()),
            ('Best AUC Top-10', ss_df['AUC_Top10_Mean'].max(), it_df['AUC_Top10_Mean'].max()),
            ('Average Top-10 Score', ss_df['Top10_Mean'].mean(), it_df['Top10_Mean'].mean()),
            ('Best Overall Score', ss_df['Best_Score'].max(), it_df['Best_Score'].max()),
            ('Total Molecules', ss_df['Total_Molecules'].sum(), it_df['Total_Molecules'].sum()),
            ('Average Molecules/Run', ss_df['Avg_Molecules_Per_Run'].mean(), it_df['Avg_Molecules_Per_Run'].mean())
        ]

        for metric_name, ss_val, it_val in metrics:
            winner = "Single-Shot" if ss_val > it_val else "Iterative" if it_val > ss_val else "Tie"
            print(f"{metric_name:<30} {ss_val:<15.4f} {it_val:<15.4f} {winner:<15}")

        # Save summary to file
        summary_stats = {
            'Single_Shot_Summary': {
                'Total_Queries': len(ss_df),
                'Average_AUC_Top10': ss_df['AUC_Top10_Mean'].mean(),
                'Best_AUC_Top10': ss_df['AUC_Top10_Mean'].max(),
                'Average_Top10_Score': ss_df['Top10_Mean'].mean(),
                'Best_Overall_Score': ss_df['Best_Score'].max(),
                'Total_Runs': ss_df['N_Runs'].sum(),
                'Total_Molecules': ss_df['Total_Molecules'].sum(),
                'Average_Molecules_Per_Run': ss_df['Avg_Molecules_Per_Run'].mean()
            },
            'Iterative_Summary': {
                'Total_Queries': len(it_df),
                'Average_AUC_Top10': it_df['AUC_Top10_Mean'].mean(),
                'Best_AUC_Top10': it_df['AUC_Top10_Mean'].max(),
                'Average_Top10_Score': it_df['Top10_Mean'].mean(),
                'Best_Overall_Score': it_df['Best_Score'].max(),
                'Total_Runs': it_df['N_Runs'].sum(),
                'Total_Molecules': it_df['Total_Molecules'].sum(),
                'Average_Molecules_Per_Run': it_df['Avg_Molecules_Per_Run'].mean()
            }
        }

        # Save to JSON
        with open(tables_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)

        print(f"\n💾 All tables saved to: {tables_dir}")
        print(f"   - single_shot_results.csv")
        print(f"   - iterative_results.csv")
        print(f"   - single_shot_top10_formatted.txt")
        print(f"   - iterative_top10_formatted.txt")
        print(f"   - summary_statistics.json")

    def run_complete_evaluation(self):
        """Run the complete oracle evaluation pipeline with organized outputs"""
        print("🚀 Starting Complete Oracle Evaluation with TDC")
        print("=" * 80)

        # Extract experiment results
        experiment_results = self.extract_experiment_results()

        if not experiment_results:
            print("❌ No experiment results found!")
            return None

        print(f"✅ Found results for {len(experiment_results)} queries")

        # Evaluate each query
        all_evaluations = {}
        successful_evaluations = 0

        for query_name, query_results in experiment_results.items():
            print(f"\n🔍 Processing query: {query_name}")
            evaluation = self.evaluate_query_results(query_name, query_results)
            all_evaluations[query_name] = evaluation

            if evaluation is not None:
                successful_evaluations += 1

        print(f"\n✅ Successfully evaluated {successful_evaluations}/{len(experiment_results)} queries")

        self.analyze_missing_tasks(all_evaluations)

        # Create organized visualizations
        df, auc_df, top10_df = self.create_organized_visualizations(all_evaluations)

        # Create comprehensive result tables
        ss_df, it_df = self.create_comprehensive_result_tables(all_evaluations)

        # Save comprehensive results
        results_file = self.results_dir / "complete_tdc_oracle_evaluation.json"
        with open(results_file, 'w') as f:
            json.dump(all_evaluations, f, indent=2, default=str)

        print(f"\n🎉 Complete evaluation finished!")
        print(f"📊 Evaluated {successful_evaluations} queries with TDC oracles")
        print(f"📈 Generated organized visualizations in: {self.results_dir / 'visualizations'}")
        print(f"📋 Generated comprehensive tables in: {self.results_dir / 'result_tables'}")
        print(f"💾 Detailed results saved to: {results_file}")

        return all_evaluations, df, auc_df, top10_df, ss_df, it_df

def main():
    """Main execution function"""
    evaluator = ComprehensiveOracleEvaluator(results_dir="scripts/simple_query_experiment_results")

    try:
        results = evaluator.run_complete_evaluation()

        if results:
            all_evaluations, df, auc_df, top10_df, ss_df, it_df = results

            print("\n" + "=" * 80)
            print("✅ TDC ORACLE EVALUATION COMPLETE!")
            print("=" * 80)

            # Show key statistics
            if len(ss_df) > 0:
                print(f"\n🏆 Best Single-Shot Performance:")
                best_ss = ss_df.iloc[0]
                print(f"   Query: {best_ss['Query']}")
                print(f"   Oracle: {best_ss['Oracle']}")
                print(f"   AUC Top-10: {best_ss['AUC_Top10_Mean']:.4f} ± {best_ss['AUC_Top10_Std']:.4f}")
                print(f"   Best Score: {best_ss['Best_Score']:.4f}")

            if len(it_df) > 0:
                print(f"\n🚀 Best Iterative Performance:")
                best_it = it_df.iloc[0]
                print(f"   Query: {best_it['Query']}")
                print(f"   Oracle: {best_it['Oracle']}")
                print(f"   AUC Top-10: {best_it['AUC_Top10_Mean']:.4f} ± {best_it['AUC_Top10_Std']:.4f}")
                print(f"   Best Score: {best_it['Best_Score']:.4f}")

            # Coverage statistics
            print(f"\n📊 Coverage Statistics:")
            print(f"   Single-Shot Queries: {len(ss_df)}")
            print(f"   Iterative Queries: {len(it_df)}")
            print(f"   Total Evaluations: {len(ss_df) + len(it_df)}")

            return results

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()