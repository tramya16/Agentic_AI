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

    def create_top_performers_analysis(self, all_evaluations):
        """Create detailed top performers analysis with clear task identification"""
        print("\n🏆 Creating Top Performers Analysis...")

        # Create visualization directory
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Create tables directory
        tables_dir = self.results_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        # Prepare AUC data
        auc_data = []
        for query_name, eval_data in all_evaluations.items():
            if eval_data is None:
                continue

            for pipeline in ['single_shot', 'iterative']:
                pipeline_name = pipeline.replace('_', '-').title()

                if eval_data[pipeline]['auc_scores']:
                    auc_mean = np.mean(eval_data[pipeline]['auc_scores'])
                    auc_std = np.std(eval_data[pipeline]['auc_scores'])

                    auc_data.append({
                        'Query': query_name,
                        'Pipeline': pipeline_name,
                        'AUC_Mean': auc_mean,
                        'AUC_Std': auc_std,
                        'Oracle': COMPLETE_ORACLE_MAPPING[query_name]
                    })

        if not auc_data:
            print("❌ No AUC data available")
            return

        auc_df = pd.DataFrame(auc_data)

        # Get top 10 performers
        top_10 = auc_df.nlargest(10, 'AUC_Mean').reset_index(drop=True)

        # Create detailed top performers plot
        plt.figure(figsize=(16, 10))

        # Main bar plot with task names
        ax1 = plt.subplot(2, 1, 1)
        colors = ['#1f77b4' if x == 'Single-Shot' else '#ff7f0e' for x in top_10['Pipeline']]
        bars = ax1.bar(range(len(top_10)), top_10['AUC_Mean'], color=colors, alpha=0.8)

        # Add error bars
        ax1.errorbar(range(len(top_10)), top_10['AUC_Mean'],
                     yerr=top_10['AUC_Std'], fmt='none', color='black', capsize=4)

        # Add value labels on bars
        for i, (bar, row) in enumerate(zip(bars, top_10.iterrows())):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{row[1]["AUC_Mean"]:.3f}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

        # Set task names as x-labels with rotation
        task_labels = [f"{row['Query'][:15]}\n({row['Pipeline']})" for _, row in top_10.iterrows()]
        ax1.set_xticks(range(len(top_10)))
        ax1.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=9)

        ax1.set_title("Top 10 Performing Tasks (AUC Top-10)", fontsize=16, fontweight='bold')
        ax1.set_ylabel("AUC Top-10 Score", fontsize=12)
        ax1.grid(axis='y', alpha=0.3)

        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Single-Shot'),
                           Patch(facecolor='#ff7f0e', label='Iterative')]
        ax1.legend(handles=legend_elements, title='Pipeline', loc='upper right')

        # Bottom subplot: Pipeline comparison for top tasks
        ax2 = plt.subplot(2, 1, 2)

        # Get top 5 tasks (by query name, not pipeline)
        top_queries = auc_df.groupby('Query')['AUC_Mean'].max().nlargest(5)

        pipeline_comparison_data = []
        for query in top_queries.index:
            query_data = auc_df[auc_df['Query'] == query]
            for _, row in query_data.iterrows():
                pipeline_comparison_data.append({
                    'Query': query[:15],  # Truncate for display
                    'Pipeline': row['Pipeline'],
                    'AUC_Mean': row['AUC_Mean'],
                    'AUC_Std': row['AUC_Std']
                })

        comp_df = pd.DataFrame(pipeline_comparison_data)

        # Grouped bar plot
        x_pos = np.arange(len(top_queries))
        width = 0.35

        ss_data = comp_df[comp_df['Pipeline'] == 'Single-Shot']
        it_data = comp_df[comp_df['Pipeline'] == 'Iterative']

        # Align data properly
        ss_values = []
        it_values = []
        ss_stds = []
        it_stds = []

        for query in top_queries.index:
            query_short = query[:15]
            ss_row = ss_data[ss_data['Query'] == query_short]
            it_row = it_data[it_data['Query'] == query_short]

            ss_values.append(ss_row['AUC_Mean'].iloc[0] if len(ss_row) > 0 else 0)
            it_values.append(it_row['AUC_Mean'].iloc[0] if len(it_row) > 0 else 0)
            ss_stds.append(ss_row['AUC_Std'].iloc[0] if len(ss_row) > 0 else 0)
            it_stds.append(it_row['AUC_Std'].iloc[0] if len(it_row) > 0 else 0)

        bars1 = ax2.bar(x_pos - width / 2, ss_values, width, label='Single-Shot',
                        color='#1f77b4', alpha=0.8, yerr=ss_stds, capsize=4)
        bars2 = ax2.bar(x_pos + width / 2, it_values, width, label='Iterative',
                        color='#ff7f0e', alpha=0.8, yerr=it_stds, capsize=4)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, ss_values)):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        for i, (bar, val) in enumerate(zip(bars2, it_values)):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax2.set_xlabel('Top Performing Queries', fontsize=12)
        ax2.set_ylabel('AUC Top-10 Score', fontsize=12)
        ax2.set_title('Pipeline Comparison for Top 5 Queries', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([q[:15] for q in top_queries.index], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "top_performers_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Print detailed top performers table
        print("\n" + "=" * 100)
        print("🏆 TOP 10 PERFORMING TASKS (DETAILED)")
        print("=" * 100)
        print(f"{'Rank':<4} {'Task':<25} {'Oracle':<20} {'Pipeline':<12} {'AUC-10':<10} {'±Std':<8}")
        print("-" * 100)

        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i:<4} {row['Query']:<25} {row['Oracle']:<20} {row['Pipeline']:<12} "
                  f"{row['AUC_Mean']:<10.4f} ±{row['AUC_Std']:<7.4f}")

        # Show pipeline distribution in top 10
        ss_count = len(top_10[top_10['Pipeline'] == 'Single-Shot'])
        it_count = len(top_10[top_10['Pipeline'] == 'Iterative'])

        print("-" * 100)
        print(f"📊 PIPELINE DISTRIBUTION IN TOP 10:")
        print(f"   Single-Shot: {ss_count}/10 tasks ({ss_count * 10}%)")
        print(f"   Iterative: {it_count}/10 tasks ({it_count * 10}%)")

        # Show best task for each pipeline
        best_ss = auc_df[auc_df['Pipeline'] == 'Single-Shot'].nlargest(1, 'AUC_Mean')
        best_it = auc_df[auc_df['Pipeline'] == 'Iterative'].nlargest(1, 'AUC_Mean')

        print(f"\n🥇 BEST PERFORMANCE BY PIPELINE:")
        if len(best_ss) > 0:
            row = best_ss.iloc[0]
            print(f"   Single-Shot Best: {row['Query']} (AUC: {row['AUC_Mean']:.4f}±{row['AUC_Std']:.4f})")

        if len(best_it) > 0:
            row = best_it.iloc[0]
            print(f"   Iterative Best: {row['Query']} (AUC: {row['AUC_Mean']:.4f}±{row['AUC_Std']:.4f})")

        # Save top performers to text file
        with open(tables_dir / "top_10_performers.txt", 'w') as f:
            f.write("TOP 10 PERFORMING TASKS (DETAILED)\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Rank':<4} {'Task':<25} {'Oracle':<20} {'Pipeline':<12} {'AUC-10':<10} {'±Std':<8}\n")
            f.write("-" * 100 + "\n")

            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"{i:<4} {row['Query']:<25} {row['Oracle']:<20} {row['Pipeline']:<12} "
                        f"{row['AUC_Mean']:<10.4f} ±{row['AUC_Std']:<7.4f}\n")

            f.write("-" * 100 + "\n")
            f.write(f"PIPELINE DISTRIBUTION IN TOP 10:\n")
            f.write(f"   Single-Shot: {ss_count}/10 tasks ({ss_count * 10}%)\n")
            f.write(f"   Iterative: {it_count}/10 tasks ({it_count * 10}%)\n")

            if len(best_ss) > 0:
                row = best_ss.iloc[0]
                f.write(f"\nBEST SINGLE-SHOT: {row['Query']} (AUC: {row['AUC_Mean']:.4f}±{row['AUC_Std']:.4f})\n")

            if len(best_it) > 0:
                row = best_it.iloc[0]
                f.write(f"BEST ITERATIVE: {row['Query']} (AUC: {row['AUC_Mean']:.4f}±{row['AUC_Std']:.4f})\n")

        print(f"\n💾 Top performers analysis saved to: {viz_dir / 'top_performers_detailed.png'}")
        print(f"💾 Top performers table saved to: {tables_dir / 'top_10_performers.txt'}")

        return top_10

    def create_simple_visualizations(self, all_evaluations):
        """Create comprehensive visualizations in a single folder"""
        print("\n📊 Creating visualizations...")

        # Create single visualization directory
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

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

        # Create main comprehensive plot
        fig = plt.figure(figsize=(20, 16))

        # 1. Main AUC Results
        ax1 = plt.subplot(3, 3, 1)
        auc_summary = auc_df.groupby(['Query', 'Pipeline']).agg({
            'AUC_Top10': ['mean', 'std']
        }).round(4)
        auc_summary.columns = ['AUC_Mean', 'AUC_Std']
        auc_summary = auc_summary.reset_index()

        sns.barplot(data=auc_summary, x='Query', y='AUC_Mean', hue='Pipeline', ax=ax1)
        ax1.set_title("AUC Top-10 Performance", fontsize=14, fontweight='bold')
        ax1.set_ylabel("AUC Top-10")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # 2. Pipeline Comparison
        ax2 = plt.subplot(3, 3, 2)
        sns.violinplot(data=auc_df, x='Pipeline', y='AUC_Top10', ax=ax2)
        ax2.set_title("AUC Distribution by Pipeline", fontweight='bold')

        # 3. Success Rate
        ax3 = plt.subplot(3, 3, 3)
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
        sns.boxplot(data=success_df, x='Pipeline', y='Success_Rate', ax=ax3)
        ax3.set_title("Success Rate (Score > 0.8)", fontweight='bold')
        ax3.set_ylabel("Success Rate (%)")

        # 4. MW vs Oracle Score
        ax4 = plt.subplot(3, 3, 4)
        scatter = ax4.scatter(df['MW'], df['Oracle_Score'], c=df['LogP'], cmap='viridis', alpha=0.6, s=15)
        ax4.set_xlabel("Molecular Weight")
        ax4.set_ylabel("Oracle Score")
        ax4.set_title("MW vs Oracle Score (colored by LogP)")
        plt.colorbar(scatter, ax=ax4, label='LogP')

        # 5. LogP vs Oracle Score
        ax5 = plt.subplot(3, 3, 5)
        scatter2 = ax5.scatter(df['LogP'], df['Oracle_Score'], c=df['TPSA'], cmap='plasma', alpha=0.6, s=15)
        ax5.set_xlabel("LogP")
        ax5.set_ylabel("Oracle Score")
        ax5.set_title("LogP vs Oracle Score (colored by TPSA)")
        plt.colorbar(scatter2, ax=ax5, label='TPSA')

        # 6. Query Heatmap
        ax6 = plt.subplot(3, 3, 6)
        heatmap_data = auc_df.pivot_table(values='AUC_Top10', index='Query', columns='Pipeline', aggfunc='mean')
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=ax6)
        ax6.set_title("Query Performance Heatmap")

        # 7. Score Distribution
        ax7 = plt.subplot(3, 3, 7)
        for pipeline in df['Pipeline'].unique():
            subset = df[df['Pipeline'] == pipeline]
            ax7.hist(subset['Oracle_Score'], alpha=0.6, label=pipeline, bins=30, density=True)
        ax7.set_xlabel("Oracle Score")
        ax7.set_ylabel("Density")
        ax7.set_title("Score Distribution")
        ax7.legend()

        # 8. FIXED Top Performers with task labels
        ax8 = plt.subplot(3, 3, 8)
        top_10_tasks = auc_summary.nlargest(10, 'AUC_Mean').reset_index(drop=True)
        colors = ['#1f77b4' if x == 'Single-Shot' else '#ff7f0e' for x in top_10_tasks['Pipeline']]
        bars = ax8.bar(range(len(top_10_tasks)), top_10_tasks['AUC_Mean'], color=colors, alpha=0.8)

        # Add task labels with rotation
        task_labels = [f"{row['Query'][:10]}" for _, row in top_10_tasks.iterrows()]
        ax8.set_xticks(range(len(top_10_tasks)))
        ax8.set_xticklabels(task_labels, rotation=45, ha='right', fontsize=8)

        # Add value labels on bars
        for i, (bar, row) in enumerate(zip(bars, top_10_tasks.iterrows())):
            ax8.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{row[1]["AUC_Mean"]:.3f}', ha='center', va='bottom', fontsize=8)

        ax8.set_title("Top 10 Performers (with task names)", fontsize=12, fontweight='bold')
        ax8.set_ylabel("AUC Top-10")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Single-Shot'),
                           Patch(facecolor='#ff7f0e', label='Iterative')]
        ax8.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # 9. Efficiency Analysis
        ax9 = plt.subplot(3, 3, 9)
        efficiency_data = auc_df.groupby('Pipeline').agg({
            'AUC_Top10': 'mean',
            'N_Molecules': 'mean'
        }).reset_index()

        for i, (_, row) in enumerate(efficiency_data.iterrows()):
            ax9.scatter(row['N_Molecules'], row['AUC_Top10'], s=300, alpha=0.7, label=row['Pipeline'])
            ax9.annotate(row['Pipeline'], (row['N_Molecules'], row['AUC_Top10']),
                         xytext=(10, 10), textcoords='offset points', fontweight='bold')

        ax9.set_xlabel("Average Molecules Generated")
        ax9.set_ylabel("Average AUC Top-10")
        ax9.set_title("Pipeline Efficiency")
        ax9.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "comprehensive_oracle_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create detailed top performers analysis
        self.create_top_performers_analysis(all_evaluations)

        print(f"📊 Visualizations saved to: {viz_dir}")
        return df, auc_df, top10_df

    def create_latex_tables(self, all_evaluations):
        """Create LaTeX tables with corrected Best Score calculations"""
        print("\n📋 Creating LaTeX tables...")

        # Create tables directory
        tables_dir = self.results_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        # Get all possible tasks
        all_possible_tasks = set(COMPLETE_ORACLE_MAPPING.keys())

        # Prepare data for tables
        single_shot_results = []
        iterative_results = []

        for task_name in sorted(all_possible_tasks):
            eval_data = all_evaluations.get(task_name)
            oracle_name = COMPLETE_ORACLE_MAPPING[task_name]

            if eval_data is None:
                # Task failed - add zeros
                single_shot_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': 0.0,
                    'AUC_Top10_Std': 0.0,
                    'Top10_Mean': 0.0,
                    'Top10_Std': 0.0,
                    'Best_Score': 0.0,  # Best single score across all runs
                    'N_Runs': 0,
                    'Total_Molecules': 0
                })

                iterative_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': 0.0,
                    'AUC_Top10_Std': 0.0,
                    'Top10_Mean': 0.0,
                    'Top10_Std': 0.0,
                    'Best_Score': 0.0,  # Best single score across all runs
                    'N_Runs': 0,
                    'Total_Molecules': 0
                })
                continue

            # Single-shot results
            if eval_data["single_shot"]["auc_scores"]:
                ss_auc_scores = eval_data["single_shot"]["auc_scores"]
                ss_top10_scores = eval_data["single_shot"]["top_10_scores"]

                # CORRECTED: Find the best SINGLE molecule score across ALL runs
                all_molecule_scores = []
                for run in eval_data["single_shot"]["runs"]:
                    all_molecule_scores.extend([mol['Oracle_Score'] for mol in run['molecules']])

                best_single_score = max(all_molecule_scores) if all_molecule_scores else 0.0

                ss_total_molecules = sum(len(run["molecules"]) for run in eval_data["single_shot"]["runs"])

                single_shot_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': np.mean(ss_auc_scores),
                    'AUC_Top10_Std': np.std(ss_auc_scores),
                    'Top10_Mean': np.mean(ss_top10_scores),
                    'Top10_Std': np.std(ss_top10_scores),
                    'Best_Score': best_single_score,  # Best single molecule score
                    'N_Runs': len(ss_auc_scores),
                    'Total_Molecules': ss_total_molecules
                })
            else:
                single_shot_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': 0.0,
                    'AUC_Top10_Std': 0.0,
                    'Top10_Mean': 0.0,
                    'Top10_Std': 0.0,
                    'Best_Score': 0.0,
                    'N_Runs': 0,
                    'Total_Molecules': 0
                })

            # Iterative results
            if eval_data["iterative"]["auc_scores"]:
                it_auc_scores = eval_data["iterative"]["auc_scores"]
                it_top10_scores = eval_data["iterative"]["top_10_scores"]

                # CORRECTED: Find the best SINGLE molecule score across ALL runs
                all_molecule_scores = []
                for run in eval_data["iterative"]["runs"]:
                    all_molecule_scores.extend([mol['Oracle_Score'] for mol in run['molecules']])

                best_single_score = max(all_molecule_scores) if all_molecule_scores else 0.0

                it_total_molecules = sum(len(run["molecules"]) for run in eval_data["iterative"]["runs"])

                iterative_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': np.mean(it_auc_scores),
                    'AUC_Top10_Std': np.std(it_auc_scores),
                    'Top10_Mean': np.mean(it_top10_scores),
                    'Top10_Std': np.std(it_top10_scores),
                    'Best_Score': best_single_score,  # Best single molecule score
                    'N_Runs': len(it_auc_scores),
                    'Total_Molecules': it_total_molecules
                })
            else:
                iterative_results.append({
                    'Query': task_name,
                    'Oracle': oracle_name,
                    'AUC_Top10_Mean': 0.0,
                    'AUC_Top10_Std': 0.0,
                    'Top10_Mean': 0.0,
                    'Top10_Std': 0.0,
                    'Best_Score': 0.0,
                    'N_Runs': 0,
                    'Total_Molecules': 0
                })

        # Create DataFrames and sort by AUC
        ss_df = pd.DataFrame(single_shot_results).sort_values('AUC_Top10_Mean', ascending=False)
        it_df = pd.DataFrame(iterative_results).sort_values('AUC_Top10_Mean', ascending=False)

        # Create LaTeX tables
        self._create_latex_table(ss_df, "Single-Shot Pipeline", tables_dir / "single_shot_results.tex")
        self._create_latex_table(it_df, "Iterative Pipeline", tables_dir / "iterative_results.tex")

        # Create combined comparison table
        self._create_combined_latex_table(ss_df, it_df, tables_dir / "combined_results.tex")

        # NEW: Create task-wise comparison table (simplified without Oracle column)
        self._create_taskwise_comparison_table(ss_df, it_df, tables_dir / "taskwise_comparison.tex")

        # Print summary with corrected calculations
        self._print_corrected_summary(ss_df, it_df)

        return ss_df, it_df

    def _create_latex_table(self, df, title, filename):
        """Create a LaTeX table"""
        latex_content = f"""\\begin{{table}}[h!]
\\centering
\\caption{{{title} - TDC Oracle Evaluation Results}}
\\label{{tab:{title.lower().replace(' ', '_')}}}
\\begin{{tabular}}{{|l|l|c|c|c|c|c|c|}}
\\hline
\\textbf{{Query}} & \\textbf{{Oracle}} & \\textbf{{AUC-10}} & \\textbf{{±Std}} & \\textbf{{Top-10}} & \\textbf{{Best}} & \\textbf{{Runs}} & \\textbf{{Mols}} \\\\
\\hline
"""

        for _, row in df.iterrows():
            query_short = row['Query'].replace('_', '\\_')[:20]
            oracle_short = row['Oracle'].replace('_', '\\_')[:15]

            latex_content += f"{query_short} & {oracle_short} & {row['AUC_Top10_Mean']:.4f} & ±{row['AUC_Top10_Std']:.4f} & {row['Top10_Mean']:.4f} & {row['Best_Score']:.4f} & {row['N_Runs']} & {row['Total_Molecules']} \\\\\n"

        # Add summary row
        auc_sum = df['AUC_Top10_Mean'].sum()
        auc_mean = df['AUC_Top10_Mean'].mean()
        auc_std = df['AUC_Top10_Mean'].std()
        top10_sum = df['Top10_Mean'].sum()
        best_overall = df['Best_Score'].max()
        total_runs = df['N_Runs'].sum()
        total_molecules = df['Total_Molecules'].sum()

        latex_content += f"""\\hline
\\textbf{{TOTAL ({len(df)} tasks)}} & \\textbf{{ALL TASKS}} & \\textbf{{{auc_sum:.4f}}} & \\textbf{{±{auc_std:.4f}}} & \\textbf{{{top10_sum:.4f}}} & \\textbf{{{best_overall:.4f}}} & \\textbf{{{total_runs}}} & \\textbf{{{total_molecules}}} \\\\
\\textbf{{AVERAGE per task}} & \\textbf{{MEAN}} & \\textbf{{{auc_mean:.4f}}} & \\textbf{{±{auc_std:.4f}}} & \\textbf{{{top10_sum / len(df):.4f}}} & \\textbf{{{best_overall:.4f}}} & \\textbf{{{total_runs / len(df):.1f}}} & \\textbf{{{total_molecules / len(df):.1f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""

        with open(filename, 'w') as f:
            f.write(latex_content)

        print(f"📄 LaTeX table saved: {filename}")

    def _create_combined_latex_table(self, ss_df, it_df, filename):
        """Create a combined comparison LaTeX table"""
        latex_content = """\\begin{table}[h!]
\\centering
\\caption{Pipeline Comparison - TDC Oracle Evaluation Summary}
\\label{tab:pipeline_comparison}
\\begin{tabular}{|l|c|c|c|c|c|c|c|}
\\hline
\\textbf{Pipeline} & \\textbf{Tasks} & \\textbf{AUC Sum} & \\textbf{AUC Avg} & \\textbf{Top-10 Sum} & \\textbf{Best} & \\textbf{Runs} & \\textbf{Mols} \\\\
\\hline
"""

        # Single-shot summary
        ss_auc_sum = ss_df['AUC_Top10_Mean'].sum()
        ss_auc_mean = ss_df['AUC_Top10_Mean'].mean()
        ss_top10_sum = ss_df['Top10_Mean'].sum()
        ss_best = ss_df['Best_Score'].max()
        ss_runs = ss_df['N_Runs'].sum()
        ss_molecules = ss_df['Total_Molecules'].sum()

        # Iterative summary
        it_auc_sum = it_df['AUC_Top10_Mean'].sum()
        it_auc_mean = it_df['AUC_Top10_Mean'].mean()
        it_top10_sum = it_df['Top10_Mean'].sum()
        it_best = it_df['Best_Score'].max()
        it_runs = it_df['N_Runs'].sum()
        it_molecules = it_df['Total_Molecules'].sum()

        latex_content += f"""Single-Shot & {len(ss_df)} & {ss_auc_sum:.4f} & {ss_auc_mean:.4f} & {ss_top10_sum:.4f} & {ss_best:.4f} & {ss_runs} & {ss_molecules} \\\\
Iterative & {len(it_df)} & {it_auc_sum:.4f} & {it_auc_mean:.4f} & {it_top10_sum:.4f} & {it_best:.4f} & {it_runs} & {it_molecules} \\\\
\\hline
"""

        # Differences
        auc_sum_diff = it_auc_sum - ss_auc_sum
        auc_mean_diff = it_auc_mean - ss_auc_mean
        top10_sum_diff = it_top10_sum - ss_top10_sum
        runs_diff = it_runs - ss_runs
        molecules_diff = it_molecules - ss_molecules

        latex_content += f"""\\textbf{{Difference}} & {len(it_df) - len(ss_df):+} & \\textbf{{{auc_sum_diff:+.4f}}} & \\textbf{{{auc_mean_diff:+.4f}}} & \\textbf{{{top10_sum_diff:+.4f}}} & = & \\textbf{{{runs_diff:+}}} & \\textbf{{{molecules_diff:+}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""

        with open(filename, 'w') as f:
            f.write(latex_content)

        print(f"📄 Combined LaTeX table saved: {filename}")

    def _create_taskwise_comparison_table(self, ss_df, it_df, filename):
        """Create a task-wise comparison LaTeX table with best scores in bold (simplified without Oracle column)"""
        print("\n📊 Creating task-wise comparison table...")

        # Merge dataframes on Query name
        comparison_data = []

        # Get all unique tasks
        all_tasks = set(ss_df['Query'].tolist() + it_df['Query'].tolist())

        for task in sorted(all_tasks):
            # Get single-shot data
            ss_row = ss_df[ss_df['Query'] == task]
            ss_auc = ss_row['AUC_Top10_Mean'].iloc[0] if len(ss_row) > 0 else 0.0
            ss_top10 = ss_row['Top10_Mean'].iloc[0] if len(ss_row) > 0 else 0.0
            ss_best = ss_row['Best_Score'].iloc[0] if len(ss_row) > 0 else 0.0

            # Get iterative data
            it_row = it_df[it_df['Query'] == task]
            it_auc = it_row['AUC_Top10_Mean'].iloc[0] if len(it_row) > 0 else 0.0
            it_top10 = it_row['Top10_Mean'].iloc[0] if len(it_row) > 0 else 0.0
            it_best = it_row['Best_Score'].iloc[0] if len(it_row) > 0 else 0.0

            comparison_data.append({
                'Task': task,
                'SS_AUC': ss_auc,
                'IT_AUC': it_auc,
                'SS_Top10': ss_top10,
                'IT_Top10': it_top10,
                'SS_Best': ss_best,
                'IT_Best': it_best
            })

        # Sort by best overall AUC score
        comparison_data.sort(key=lambda x: max(x['SS_AUC'], x['IT_AUC']), reverse=True)

        # Create LaTeX table
        latex_content = """\\begin{table}[h!]
    \\centering
    \\caption{Task-wise Comparison: Single-Shot vs Iterative Pipeline Performance}
    \\label{tab:taskwise_comparison}
    \\resizebox{\\textwidth}{!}{%
    \\begin{tabular}{|l|c|c|c|c|c|c|c|}
    \\hline
    \\multirow{2}{*}{\\textbf{Task}} & \\multicolumn{2}{c|}{\\textbf{AUC-10}} & \\multicolumn{2}{c|}{\\textbf{Top-10 Mean}} & \\multicolumn{2}{c|}{\\textbf{Best Score}} & \\textbf{Winner} \\\\
    \\cline{2-7}
    & \\textbf{SS} & \\textbf{IT} & \\textbf{SS} & \\textbf{IT} & \\textbf{SS} & \\textbf{IT} & \\textbf{AUC} \\\\
    \\hline
    """

        # Track wins
        ss_wins = 0
        it_wins = 0
        ties = 0

        # Define tolerance for tie detection (adjust as needed)
        TOLERANCE = 1e-6

        for data in comparison_data:
            task_name = data['Task'].replace('_', '\\_')

            # Determine winner for AUC with tolerance for ties
            auc_diff = abs(data['SS_AUC'] - data['IT_AUC'])

            if auc_diff <= TOLERANCE:
                # It's a tie
                winner = "Tie"
                ties += 1
                ss_auc_str = f"{data['SS_AUC']:.4f}"
                it_auc_str = f"{data['IT_AUC']:.4f}"
            elif data['SS_AUC'] > data['IT_AUC']:
                # Single-shot wins
                winner = "SS"
                ss_wins += 1
                ss_auc_str = f"\\textbf{{{data['SS_AUC']:.4f}}}"
                it_auc_str = f"{data['IT_AUC']:.4f}"
            else:
                # Iterative wins
                winner = "IT"
                it_wins += 1
                ss_auc_str = f"{data['SS_AUC']:.4f}"
                it_auc_str = f"\\textbf{{{data['IT_AUC']:.4f}}}"

            # Bold best Top-10 scores with tolerance
            top10_diff = abs(data['SS_Top10'] - data['IT_Top10'])

            if top10_diff <= TOLERANCE:
                # Tie in Top-10
                ss_top10_str = f"{data['SS_Top10']:.4f}"
                it_top10_str = f"{data['IT_Top10']:.4f}"
            elif data['SS_Top10'] > data['IT_Top10']:
                ss_top10_str = f"\\textbf{{{data['SS_Top10']:.4f}}}"
                it_top10_str = f"{data['IT_Top10']:.4f}"
            else:
                ss_top10_str = f"{data['SS_Top10']:.4f}"
                it_top10_str = f"\\textbf{{{data['IT_Top10']:.4f}}}"

            # Bold best individual scores with tolerance
            best_diff = abs(data['SS_Best'] - data['IT_Best'])

            if best_diff <= TOLERANCE:
                # Tie in Best scores
                ss_best_str = f"{data['SS_Best']:.4f}"
                it_best_str = f"{data['IT_Best']:.4f}"
            elif data['SS_Best'] > data['IT_Best']:
                ss_best_str = f"\\textbf{{{data['SS_Best']:.4f}}}"
                it_best_str = f"{data['IT_Best']:.4f}"
            else:
                ss_best_str = f"{data['SS_Best']:.4f}"
                it_best_str = f"\\textbf{{{data['IT_Best']:.4f}}}"

            latex_content += f"""{task_name} & {ss_auc_str} & {it_auc_str} & {ss_top10_str} & {it_top10_str} & {ss_best_str} & {it_best_str} & {winner} \\\\
    """

        # Add summary statistics
        total_tasks = len(comparison_data)
        ss_win_pct = (ss_wins / total_tasks) * 100 if total_tasks > 0 else 0
        it_win_pct = (it_wins / total_tasks) * 100 if total_tasks > 0 else 0
        tie_pct = (ties / total_tasks) * 100 if total_tasks > 0 else 0

        # Calculate total scores
        total_ss_auc = sum(d['SS_AUC'] for d in comparison_data)
        total_it_auc = sum(d['IT_AUC'] for d in comparison_data)
        total_ss_top10 = sum(d['SS_Top10'] for d in comparison_data)
        total_it_top10 = sum(d['IT_Top10'] for d in comparison_data)

        latex_content += f"""\\hline
    \\multicolumn{{8}}{{|c|}}{{\\textbf{{SUMMARY STATISTICS}}}} \\\\
    \\hline
    \\multicolumn{{1}}{{|l|}}{{\\textbf{{Total Tasks}}}} & \\multicolumn{{7}}{{c|}}{{{total_tasks}}} \\\\
    \\multicolumn{{1}}{{|l|}}{{\\textbf{{Single-Shot Wins}}}} & \\multicolumn{{7}}{{c|}}{{{ss_wins} ({ss_win_pct:.1f}\\%)}} \\\\
    \\multicolumn{{1}}{{|l|}}{{\\textbf{{Iterative Wins}}}} & \\multicolumn{{7}}{{c|}}{{{it_wins} ({it_win_pct:.1f}\\%)}} \\\\
    \\multicolumn{{1}}{{|l|}}{{\\textbf{{Ties}}}} & \\multicolumn{{7}}{{c|}}{{{ties} ({tie_pct:.1f}\\%)}} \\\\
    \\hline
    \\multicolumn{{1}}{{|l|}}{{\\textbf{{Total AUC}}}} & {total_ss_auc:.4f} & {total_it_auc:.4f} & \\multicolumn{{5}}{{c|}}{{Difference: {total_it_auc - total_ss_auc:+.4f}}} \\\\
    \\multicolumn{{1}}{{|l|}}{{\\textbf{{Total Top-10}}}} & \\multicolumn{{2}}{{c|}}{{N/A}} & {total_ss_top10:.4f} & {total_it_top10:.4f} & \\multicolumn{{3}}{{c|}}{{Difference: {total_it_top10 - total_ss_top10:+.4f}}} \\\\
    \\hline
    \\end{{tabular}}%
    }}
    \\end{{table}}
    """

        with open(filename, 'w') as f:
            f.write(latex_content)

        # Print summary to console with detailed breakdown
        print(f"\n📊 TASK-WISE COMPARISON SUMMARY:")
        print(f"   Total Tasks: {total_tasks}")
        print(f"   Single-Shot Wins: {ss_wins} ({ss_win_pct:.1f}%)")
        print(f"   Iterative Wins: {it_wins} ({it_win_pct:.1f}%)")
        print(f"   Ties: {ties} ({tie_pct:.1f}%)")
        print(
            f"   Total AUC - SS: {total_ss_auc:.4f}, IT: {total_it_auc:.4f}, Diff: {total_it_auc - total_ss_auc:+.4f}")
        print(
            f"   Total Top-10 - SS: {total_ss_top10:.4f}, IT: {total_it_top10:.4f}, Diff: {total_it_top10 - total_ss_top10:+.4f}")

        # Print detailed breakdown of ties for debugging
        print(f"\n🔍 DETAILED TIE ANALYSIS:")
        tie_tasks = []
        for data in comparison_data:
            auc_diff = abs(data['SS_AUC'] - data['IT_AUC'])
            if auc_diff <= TOLERANCE:
                tie_tasks.append(f"{data['Task']} (AUC diff: {auc_diff:.6f})")

        if tie_tasks:
            print(f"   Tasks with AUC ties (tolerance: {TOLERANCE}):")
            for tie_task in tie_tasks:
                print(f"     - {tie_task}")
        else:
            print(f"   No AUC ties found (tolerance: {TOLERANCE})")

        print(f"📄 Task-wise comparison table saved: {filename}")

        return comparison_data

    def _print_corrected_summary(self, ss_df, it_df):
        """Print corrected summary with proper Best Score explanation"""
        print("\n" + "=" * 120)
        print("🏆 CORRECTED FINAL RESULTS SUMMARY")
        print("=" * 120)

        print(f"\n📊 METRIC EXPLANATIONS:")
        print(f"   • AUC-10: Area Under Curve for top-10 molecules (higher = better)")
        print(f"   • Top-10: Average score of top-10 molecules per run")
        print(f"   • Best: Highest single molecule score across ALL runs for each task")
        print(f"   • Sum: Total of all task scores (NOT average)")

        # Single-shot summary
        ss_auc_sum = ss_df['AUC_Top10_Mean'].sum()
        ss_auc_mean = ss_df['AUC_Top10_Mean'].mean()
        ss_top10_sum = ss_df['Top10_Mean'].sum()
        ss_best_sum = ss_df['Best_Score'].sum()  # Sum of best scores
        ss_best_max = ss_df['Best_Score'].max()  # Maximum best score
        ss_runs = ss_df['N_Runs'].sum()
        ss_molecules = ss_df['Total_Molecules'].sum()

        # Iterative summary
        it_auc_sum = it_df['AUC_Top10_Mean'].sum()
        it_auc_mean = it_df['AUC_Top10_Mean'].mean()
        it_top10_sum = it_df['Top10_Mean'].sum()
        it_best_sum = it_df['Best_Score'].sum()  # Sum of best scores
        it_best_max = it_df['Best_Score'].max()  # Maximum best score
        it_runs = it_df['N_Runs'].sum()
        it_molecules = it_df['Total_Molecules'].sum()

        print(f"\n🥇 SINGLE-SHOT PIPELINE SUMMARY:")
        print(f"   Tasks: {len(ss_df)}")
        print(f"   AUC Sum: {ss_auc_sum:.4f} (sum of all task AUC scores)")
        print(f"   AUC Mean: {ss_auc_mean:.4f} (average AUC per task)")
        print(f"   Top-10 Sum: {ss_top10_sum:.4f} (sum of all task top-10 scores)")
        print(f"   Best Score Sum: {ss_best_sum:.4f} (sum of best scores from each task)")
        print(f"   Best Score Max: {ss_best_max:.4f} (highest single molecule score)")
        print(f"   Total Runs: {ss_runs}")
        print(f"   Total Molecules: {ss_molecules}")

        print(f"\n🚀 ITERATIVE PIPELINE SUMMARY:")
        print(f"   Tasks: {len(it_df)}")
        print(f"   AUC Sum: {it_auc_sum:.4f} (sum of all task AUC scores)")
        print(f"   AUC Mean: {it_auc_mean:.4f} (average AUC per task)")
        print(f"   Top-10 Sum: {it_top10_sum:.4f} (sum of all task top-10 scores)")
        print(f"   Best Score Sum: {it_best_sum:.4f} (sum of best scores from each task)")
        print(f"   Best Score Max: {it_best_max:.4f} (highest single molecule score)")
        print(f"   Total Runs: {it_runs}")
        print(f"   Total Molecules: {it_molecules}")

        # Differences
        auc_sum_diff = it_auc_sum - ss_auc_sum
        auc_mean_diff = it_auc_mean - ss_auc_mean
        top10_sum_diff = it_top10_sum - ss_top10_sum
        best_sum_diff = it_best_sum - ss_best_sum

        print(f"\n📈 DIFFERENCES (Iterative - Single-Shot):")
        print(f"   AUC Sum Difference: {auc_sum_diff:+.4f}")
        print(f"   AUC Mean Difference: {auc_mean_diff:+.4f}")
        print(f"   Top-10 Sum Difference: {top10_sum_diff:+.4f}")
        print(f"   Best Score Sum Difference: {best_sum_diff:+.4f}")

        # Winners
        auc_sum_winner = "Iterative" if auc_sum_diff > 0 else "Single-Shot" if auc_sum_diff < 0 else "Tie"
        auc_mean_winner = "Iterative" if auc_mean_diff > 0 else "Single-Shot" if auc_mean_diff < 0 else "Tie"
        top10_winner = "Iterative" if top10_sum_diff > 0 else "Single-Shot" if top10_sum_diff < 0 else "Tie"
        best_winner = "Iterative" if best_sum_diff > 0 else "Single-Shot" if best_sum_diff < 0 else "Tie"

        print(f"\n🏆 WINNERS:")
        print(f"   AUC Sum Winner: {auc_sum_winner}")
        print(f"   AUC Mean Winner: {auc_mean_winner}")
        print(f"   Top-10 Sum Winner: {top10_winner}")
        print(f"   Best Score Sum Winner: {best_winner}")

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

        # Create visualizations (single folder) - this now includes top performers analysis
        df, auc_df, top10_df = self.create_simple_visualizations(all_evaluations)

        # Create LaTeX tables (single folder) - now includes task-wise comparison
        ss_df, it_df = self.create_latex_tables(all_evaluations)

        # Save comprehensive results
        results_file = self.results_dir / "complete_tdc_oracle_evaluation.json"
        with open(results_file, 'w') as f:
            json.dump(all_evaluations, f, indent=2, default=str)

        print(f"\n🎉 Complete evaluation finished!")
        print(f"📊 Evaluated {successful_evaluations} queries with TDC oracles")
        print(f"📈 Visualizations saved to: {self.results_dir / 'visualizations'}")
        print(f"📋 LaTeX tables saved to: {self.results_dir / 'tables'}")
        print(f"📋 Task-wise comparison table created (simplified format without Oracle column)")
        print(f"💾 Detailed results saved to: {results_file}")

        return all_evaluations, df, auc_df, top10_df, ss_df, it_df


def main():
    """Main execution function"""
    evaluator = ComprehensiveOracleEvaluator(results_dir="old_results/Gemini_2.0_Flash_experiment_results_Temp_0.9")

    try:
        results = evaluator.run_complete_evaluation()

        if results:
            all_evaluations, df, auc_df, top10_df, ss_df, it_df = results

            print("\n" + "=" * 80)
            print("✅ TDC ORACLE EVALUATION COMPLETE!")
            print("=" * 80)

            return results

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()