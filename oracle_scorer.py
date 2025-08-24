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
    "thiothixene_rediscovery": "Thiothixene_Rediscovery",
    "troglitazone_rediscovery": "Troglitazone_Rediscovery",
    "valsartan_smarts": "Valsartan_SMARTS",
    "zaleplon_similarity": "Zaleplon_MPO"
}


class ComprehensiveOracleEvaluator:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.oracles = {}
        self.load_all_oracles()

    def load_all_oracles(self):
        """Load all TDC oracles"""
        print("Loading TDC Oracle models...")

        for query_name, oracle_name in COMPLETE_ORACLE_MAPPING.items():
            try:
                oracle = Oracle(name=oracle_name)
                self.oracles[query_name] = oracle
                print(f"Loaded {oracle_name} for {query_name}")
            except Exception as e:
                print(f"Failed to load {oracle_name}: {e}")
                self.oracles[query_name] = None

        print(
            f"\nSuccessfully loaded {len([o for o in self.oracles.values() if o is not None])}/{len(COMPLETE_ORACLE_MAPPING)} oracles")

    def score_molecule(self, smiles, query_name):
        """Score a single molecule using TDC oracle"""
        if query_name not in self.oracles or self.oracles[query_name] is None:
            return 0.0

        try:
            score = self.oracles[query_name](smiles)
            return float(score) if score is not None else 0.0
        except Exception as e:
            print(f"Oracle scoring failed for {smiles}: {e}")
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
        print(f"\nExtracting results from {self.results_dir}...")
        print(self.results_dir)
        all_results = defaultdict(lambda: {"single_shot": [], "iterative": []})

        for file_path in self.results_dir.glob("*_detailed_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                query_name = data.get("query_name", "unknown")
                print(f"Processing {file_path.name} for query: {query_name}")

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
                print(f"Error processing {file_path}: {e}")

        return dict(all_results)

    def evaluate_query_results(self, query_name, query_results):
        """Evaluate results for a single query"""
        if query_name not in self.oracles or self.oracles[query_name] is None:
            print(f"No oracle available for {query_name}")
            return None

        oracle_name = COMPLETE_ORACLE_MAPPING[query_name]
        print(f"\nEvaluating {query_name} with {oracle_name}")
        print("=" * 60)

        evaluation_results = {
            "query_name": query_name,
            "oracle_name": oracle_name,
            "single_shot": {"runs": [], "auc_scores": [], "top_10_scores": []},
            "iterative": {"runs": [], "auc_scores": [], "top_10_scores": []}
        }

        # Evaluate single-shot runs
        print(f"Single-shot evaluation ({len(query_results['single_shot'])} runs):")
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
        print(f"Iterative evaluation ({len(query_results['iterative'])} runs):")
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

    def create_clean_visualizations(self, all_evaluations):
        """Create clean, focused visualizations"""
        print("\nCreating clean visualizations...")

        # Create visualization directory
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Prepare data
        plot_data = []
        auc_data = []

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

        if not plot_data:
            print("No data for visualization")
            return None, None

        df = pd.DataFrame(plot_data)
        auc_df = pd.DataFrame(auc_data)

        # 1. AUC Performance Plot with Error Bars
        self._create_auc_performance_plot(auc_df, viz_dir)
        self._create_auc_performance_plot_horizontal_fixed(auc_df,viz_dir)
        # 2. Query Performance Heatmap
        self._create_query_performance_plot(auc_df, viz_dir)

        # 3. Score Distribution Plot
        self._create_score_distribution_plot(df, viz_dir)

        # 4. Pipeline Efficiency Plot
        self._create_pipeline_efficiency_plot(auc_df, viz_dir)

        print(f"Visualizations saved to: {viz_dir}")
        return df, auc_df

    def _create_auc_performance_plot(self, auc_df, viz_dir):
        """Create a clean AUC performance plot with proper bar alignment"""
        plt.figure(figsize=(18, 10))

        # Calculate summary statistics
        auc_summary = auc_df.groupby(['Query', 'Pipeline']).agg({
            'AUC_Top10': ['mean', 'std', 'count']
        }).round(4)
        auc_summary.columns = ['AUC_Mean', 'AUC_Std', 'Count']
        auc_summary = auc_summary.reset_index()

        # Sort by average AUC across both pipelines for better visualization
        query_order = auc_summary.groupby('Query')['AUC_Mean'].mean().sort_values(ascending=False).index

        # Prepare data for plotting
        queries = list(query_order)
        n_queries = len(queries)

        # Set up positions
        x_pos = np.arange(n_queries)
        width = 0.35

        # Separate data by pipeline
        ss_data = []
        it_data = []
        ss_errors = []
        it_errors = []

        for query in queries:
            # Single-Shot data
            ss_row = auc_summary[(auc_summary['Query'] == query) & (auc_summary['Pipeline'] == 'Single-Shot')]
            if len(ss_row) > 0:
                ss_data.append(ss_row['AUC_Mean'].iloc[0])
                ss_errors.append(ss_row['AUC_Std'].iloc[0])
            else:
                ss_data.append(0.0)
                ss_errors.append(0.0)

            # Iterative data
            it_row = auc_summary[(auc_summary['Query'] == query) & (auc_summary['Pipeline'] == 'Iterative')]
            if len(it_row) > 0:
                it_data.append(it_row['AUC_Mean'].iloc[0])
                it_errors.append(it_row['AUC_Std'].iloc[0])
            else:
                it_data.append(0.0)
                it_errors.append(0.0)

        # Create the plot
        fig, ax = plt.subplots(figsize=(18, 10))

        # Create bars
        bars1 = ax.bar(x_pos - width / 2, ss_data, width, label='Single-Shot',
                       color='#ff7f0e', alpha=0.8, capsize=5)
        bars2 = ax.bar(x_pos + width / 2, it_data, width, label='Iterative',
                       color='#1f77b4', alpha=0.8, capsize=5)

        # Add error bars
        ax.errorbar(x_pos - width / 2, ss_data, yerr=ss_errors, fmt='none',
                    color='black', capsize=4, capthick=1, elinewidth=1)
        ax.errorbar(x_pos + width / 2, it_data, yerr=it_errors, fmt='none',
                    color='black', capsize=4, capthick=1, elinewidth=1)

        # Add value labels on bars (only for non-zero values)
        for i, (bar, val, err) in enumerate(zip(bars1, ss_data, ss_errors)):
            if val > 0.001:  # Only show labels for meaningful values
                ax.text(bar.get_x() + bar.get_width() / 2, val + err + 0.02,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', rotation=0)

        for i, (bar, val, err) in enumerate(zip(bars2, it_data, it_errors)):
            if val > 0.001:  # Only show labels for meaningful values
                ax.text(bar.get_x() + bar.get_width() / 2, val + err + 0.02,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', rotation=0)

        # Customize the plot
        ax.set_xlabel('Query Task', fontsize=14, fontweight='bold')
        ax.set_ylabel('AUC Top-10 Score', fontsize=14, fontweight='bold')
        ax.set_title('AUC Top-10 Performance by Query and Pipeline', fontsize=16, fontweight='bold')

        # Set x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels([q.replace('_', ' ').title()[:20] for q in queries],
                           rotation=45, ha='right', fontsize=10)

        # Add legend
        ax.legend(title='Pipeline', title_fontsize=12, fontsize=11, loc='upper right')

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Set y-axis limits to avoid cramping
        max_val = max(max(ss_data), max(it_data))
        max_err = max(max(ss_errors), max(it_errors))
        ax.set_ylim(0, max_val + max_err + 0.1)

        # Improve layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(viz_dir / "auc_performance_clean.png", dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        print(f"Clean AUC performance plot saved to: {viz_dir / 'auc_performance_clean.png'}")

    def _create_auc_performance_plot_horizontal_fixed(self, auc_df, viz_dir):
        """Create a horizontal AUC performance plot with fixed error bars and value labels outside the bars."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        if auc_df.empty:
            print("Warning: auc_df is empty. Skipping plot.")
            return

        # Calculate summary statistics
        auc_summary = auc_df.groupby(['Query', 'Pipeline']).agg({
            'AUC_Top10': ['mean', 'std', 'count']
        }).round(4)
        auc_summary.columns = ['AUC_Mean', 'AUC_Std', 'Count']
        auc_summary = auc_summary.reset_index()

        # Sort queries by average AUC
        query_order = auc_summary.groupby('Query')['AUC_Mean'].mean().sort_values(ascending=True).index
        queries = list(query_order)
        n_queries = len(queries)

        # Y positions
        y_pos = np.arange(n_queries)
        height = 0.35

        # Prepare data and errors
        ss_data, it_data = [], []
        ss_err, it_err = [], []

        for query in queries:
            ss_row = auc_summary[(auc_summary['Query'] == query) & (auc_summary['Pipeline'] == 'Single-Shot')]
            it_row = auc_summary[(auc_summary['Query'] == query) & (auc_summary['Pipeline'] == 'Iterative')]

            ss_data.append(ss_row['AUC_Mean'].iloc[0] if len(ss_row) else 0.0)
            it_data.append(it_row['AUC_Mean'].iloc[0] if len(it_row) else 0.0)
            ss_err.append(ss_row['AUC_Std'].iloc[0] if len(ss_row) else 0.0)
            it_err.append(it_row['AUC_Std'].iloc[0] if len(it_row) else 0.0)

        # Determine maximum value for axis limit safely
        max_val_ss = max([v + e for v, e in zip(ss_data, ss_err)] + [0])
        max_val_it = max([v + e for v, e in zip(it_data, it_err)] + [0])
        max_val = max(max_val_ss, max_val_it)

        # Safety check for empty or NaN values
        if np.isnan(max_val) or max_val == 0:
            x_limit = 1.0
        else:
            x_limit = min(max_val * 1.15, 1.2)  # extend 15% for labels, cap at 1.2

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8 + n_queries * 0.4))

        bars1 = ax.barh(y_pos - height / 2, ss_data, height, label='Single-Shot',
                        color='#ff7f0e', alpha=0.8)
        bars2 = ax.barh(y_pos + height / 2, it_data, height, label='Iterative',
                        color='#1f77b4', alpha=0.8)

        # Error bars
        ax.errorbar(ss_data, y_pos - height / 2, xerr=ss_err, fmt='none',
                    color='black', capsize=3, capthick=1, elinewidth=1)
        ax.errorbar(it_data, y_pos + height / 2, xerr=it_err, fmt='none',
                    color='black', capsize=3, capthick=1, elinewidth=1)

        # Add labels outside the bars
        for bar, val, err in zip(bars1, ss_data, ss_err):
            if val > 0.001:
                ax.text(val + err + 0.02, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}±{err:.3f}', ha='left', va='center',
                        fontsize=11, fontweight='bold', color='black')

        for bar, val, err in zip(bars2, it_data, it_err):
            if val > 0.001:
                ax.text(val + err + 0.02, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}±{err:.3f}', ha='left', va='center',
                        fontsize=11, fontweight='bold', color='black')

        # Customize plot
        ax.set_ylabel('Query Task', fontsize=16, fontweight='bold')
        ax.set_xlabel('AUC-10 Score', fontsize=16, fontweight='bold')
        ax.set_title('AUC-10 Performance by Query and Pipeline', fontsize=18, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([q.replace('_', ' ').title() for q in queries], fontsize=13)
        ax.legend(title='Workflow', title_fontsize=14, fontsize=13, loc='lower right')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Remove closed borders (top & right spines)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set extended x-axis to accommodate labels safely
        ax.set_xlim(0, x_limit)

        # Padding & layout
        plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08)
        plt.tight_layout()

        # Save figure
        plt.savefig(viz_dir / "auc_performance_horizontal_fixed.png", dpi=350,
                    bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
        plt.close()

        print(f"Horizontal AUC performance plot saved to: {viz_dir / 'auc_performance_horizontal_fixed.png'}")

    def _create_query_performance_plot(self, auc_df, viz_dir):
        """Create query performance heatmap"""
        plt.figure(figsize=(14, 8))

        # Check if we have data for both pipelines
        if len(auc_df['Pipeline'].unique()) < 2:
            print("Warning: Not enough pipeline data for heatmap")
            return

        # Create pivot table for heatmap with proper handling of missing values
        heatmap_data = auc_df.pivot_table(
            values='AUC_Top10',
            index='Query',
            columns='Pipeline',
            aggfunc='mean',
            fill_value=0  # Fill missing values with 0
        )

        # Ensure both pipeline columns exist
        for pipeline in ['Single-Shot', 'Iterative']:
            if pipeline not in heatmap_data.columns:
                heatmap_data[pipeline] = 0

        # Sort by best overall performance
        heatmap_data['Overall'] = heatmap_data.mean(axis=1)
        heatmap_data = heatmap_data.sort_values('Overall', ascending=False)
        heatmap_data = heatmap_data.drop('Overall', axis=1)

        # Create a mask for values that are 0 (no data)
        mask = heatmap_data == 0

        # Create heatmap with better formatting
        fig, ax = plt.subplots(figsize=(14, max(8, len(heatmap_data) * 0.4)))

        # Use a diverging colormap that makes 0 values stand out
        cmap = sns.color_palette("viridis", as_cmap=True)

        # Plot the heatmap
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt='.3f',
                    cmap=cmap,
                    cbar_kws={'label': 'AUC Top-10 Score'},
                    square=False,
                    mask=mask,
                    ax=ax,
                    annot_kws={'size': 10, 'weight': 'bold'})

        # Add a different color for missing values
        if mask.any().any():
            # Add text for missing values
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    if mask.iloc[i, j]:
                        ax.text(j + 0.5, i + 0.5, 'N/A',
                                ha='center', va='center',
                                fontsize=9, style='italic', color='gray')

        plt.title('Query Performance Heatmap (AUC Top-10)', fontsize=16, fontweight='bold')
        plt.xlabel('Pipeline', fontsize=12)
        plt.ylabel('Query Task', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Adjust layout to prevent cutting off labels
        plt.tight_layout()

        plt.savefig(viz_dir / "query_performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Query performance heatmap saved to: {viz_dir / 'query_performance_heatmap.png'}")


    def _create_score_distribution_plot(self, df, viz_dir):
        """Create score distribution plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall score distribution by pipeline
        for pipeline in df['Pipeline'].unique():
            subset = df[df['Pipeline'] == pipeline]
            ax1.hist(subset['Oracle_Score'], alpha=0.7, label=pipeline,
                     bins=50, density=True, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Oracle Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Score Distribution by Pipeline', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Box plot of scores by pipeline
        sns.boxplot(data=df, x='Pipeline', y='Oracle_Score', ax=ax2)
        ax2.set_title('Score Distribution Box Plot', fontweight='bold')
        ax2.set_ylabel('Oracle Score')
        ax2.grid(alpha=0.3)

        # 3. Top queries score distribution
        top_queries = df.groupby('Query')['Oracle_Score'].mean().nlargest(6).index
        top_df = df[df['Query'].isin(top_queries)]

        sns.violinplot(data=top_df, x='Query', y='Oracle_Score', ax=ax3)
        ax3.set_title('Score Distribution for Top 6 Queries', fontweight='bold')
        ax3.set_xlabel('Query Task')
        ax3.set_ylabel('Oracle Score')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        # 4. High-score analysis
        high_score_threshold = df['Oracle_Score'].quantile(0.9)
        high_score_counts = df[df['Oracle_Score'] >= high_score_threshold].groupby(
            ['Query', 'Pipeline']).size().reset_index(name='Count')

        if len(high_score_counts) > 0:
            sns.barplot(data=high_score_counts, x='Query', y='Count', hue='Pipeline', ax=ax4)
            ax4.set_title(f'High Score Count (≥{high_score_threshold:.2f}) by Query', fontweight='bold')
            ax4.set_xlabel('Query Task')
            ax4.set_ylabel('Count of High Scoring Molecules')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'No high-scoring molecules found',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('High Score Analysis', fontweight='bold')

        plt.tight_layout()
        plt.savefig(viz_dir / "score_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_pipeline_efficiency_plot(self, auc_df, viz_dir):
        """Simplified pipeline efficiency analysis with 2 clean, readable plots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.4})

        # Consistent pipeline colors
        colors = {'Single-Shot': '#ff7f0e', 'Iterative': '#1f77b4'}

        # === 1. Efficiency scatter plot ===
        efficiency_data = auc_df.groupby('Pipeline').agg({
            'AUC_Top10': 'mean',
            'N_Molecules': 'mean'
        }).reset_index()

        for _, row in efficiency_data.iterrows():
            ax1.scatter(row['N_Molecules'], row['AUC_Top10'],
                        s=500, alpha=0.8, color=colors[row['Pipeline']],
                        edgecolor='black', linewidth=1.4)

            ax1.annotate(f"{row['Pipeline']}\nAUC={row['AUC_Top10']:.3f}",
                         (row['N_Molecules'], row['AUC_Top10']),
                         xytext=(12, 10), textcoords='offset points',
                         fontweight='bold', fontsize=14)

        ax1.set_xlabel('Average Molecules Generated', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Average AUC-10', fontsize=15, fontweight='bold')
        ax1.set_title('Workflow Efficiency: Performance vs Output', fontsize=17, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=13)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Custom legend for scatter plot
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[p], markeredgecolor='black',
                              markersize=14, label=p)
                   for p in efficiency_data['Pipeline']]
        ax1.legend(handles=handles, title="Workflow",
                   fontsize=13, title_fontsize=14,
                   loc='upper left', bbox_to_anchor=(1, 1))

        fig.add_artist(
            plt.Line2D([0, 1], [0.5, 0.5], color='gray', linewidth=1.5, transform=fig.transFigure, figure=fig,
                       clip_on=False))

        # === 2. Success rate analysis ===
        success_data = []
        thresholds = [0.5, 0.7, 0.8, 0.9]

        for threshold in thresholds:
            for pipeline in auc_df['Pipeline'].unique():
                pipeline_data = auc_df[auc_df['Pipeline'] == pipeline]
                total_runs = len(pipeline_data)
                successful_runs = len(pipeline_data[pipeline_data['Max_Score'] >= threshold])
                success_rate = successful_runs / total_runs * 100 if total_runs > 0 else 0

                success_data.append({
                    'Threshold': f'≥{threshold}',
                    'Pipeline': pipeline,
                    'Success_Rate': success_rate
                })

        success_df = pd.DataFrame(success_data)

        bars = sns.barplot(
            data=success_df, x='Threshold', y='Success_Rate',
            hue='Pipeline', ax=ax2,
            palette=colors, alpha=0.9
        )

        # Add value labels on bars
        for container in ax2.containers:
            ax2.bar_label(container, fmt="%.1f%%", label_type='edge',
                          fontsize=12, padding=3, fontweight='bold')

        ax2.set_title('Success Rate by Score Threshold', fontsize=17, fontweight='bold')
        ax2.set_ylabel('Success Rate (%)', fontsize=15, fontweight='bold')
        ax2.set_xlabel('Score Threshold', fontsize=15, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=13)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Move barplot legend outside for clarity
        ax2.legend(title="Workflow", fontsize=13, title_fontsize=14, bbox_to_anchor=(1, 1), loc='upper left')

        # Save figure
        plt.tight_layout()
        plt.savefig(viz_dir / "pipeline_efficiency_simplified_large.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(
            f"Simplified pipeline efficiency analysis saved to: {viz_dir / 'pipeline_efficiency_simplified_large.png'}")

    def create_top_performers_analysis(self, all_evaluations):
        """Create detailed top performers analysis with clear task identification"""
        print("\nCreating Top Performers Analysis...")

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
            print("No AUC data available")
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
        print("TOP 10 PERFORMING TASKS (DETAILED)")
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
        print(f"PIPELINE DISTRIBUTION IN TOP 10:")
        print(f"   Single-Shot: {ss_count}/10 tasks ({ss_count * 10}%)")
        print(f"   Iterative: {it_count}/10 tasks ({it_count * 10}%)")

        # Show best task for each pipeline
        best_ss = auc_df[auc_df['Pipeline'] == 'Single-Shot'].nlargest(1, 'AUC_Mean')
        best_it = auc_df[auc_df['Pipeline'] == 'Iterative'].nlargest(1, 'AUC_Mean')

        print(f"\nBEST PERFORMANCE BY PIPELINE:")
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

        print(f"\nTop performers analysis saved to: {viz_dir / 'top_performers_detailed.png'}")
        print(f"Top performers table saved to: {tables_dir / 'top_10_performers.txt'}")

        return top_10

    def create_latex_tables(self, all_evaluations):
        """Create LaTeX tables with corrected Best Score calculations"""
        print("\nCreating LaTeX tables...")

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

        print(f"LaTeX table saved: {filename}")

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

        print(f"Combined LaTeX table saved: {filename}")

    def _create_taskwise_comparison_table(self, ss_df, it_df, filename):
        """Create a task-wise comparison LaTeX table with best scores in bold (simplified without Oracle column)"""
        print("\nCreating task-wise comparison table...")

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
        print(f"\nTASK-WISE COMPARISON SUMMARY:")
        print(f"   Total Tasks: {total_tasks}")
        print(f"   Single-Shot Wins: {ss_wins} ({ss_win_pct:.1f}%)")
        print(f"   Iterative Wins: {it_wins} ({it_win_pct:.1f}%)")
        print(f"   Ties: {ties} ({tie_pct:.1f}%)")
        print(
            f"   Total AUC - SS: {total_ss_auc:.4f}, IT: {total_it_auc:.4f}, Diff: {total_it_auc - total_ss_auc:+.4f}")
        print(
            f"   Total Top-10 - SS: {total_ss_top10:.4f}, IT: {total_it_top10:.4f}, Diff: {total_it_top10 - total_ss_top10:+.4f}")

        print(f"Task-wise comparison table saved: {filename}")

        return comparison_data

    def _print_corrected_summary(self, ss_df, it_df):
        """Print corrected summary with proper Best Score explanation"""
        print("\n" + "=" * 120)
        print("CORRECTED FINAL RESULTS SUMMARY")
        print("=" * 120)

        print(f"\nMETRIC EXPLANATIONS:")
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

        print(f"\nSINGLE-SHOT PIPELINE SUMMARY:")
        print(f"   Tasks: {len(ss_df)}")
        print(f"   AUC Sum: {ss_auc_sum:.4f} (sum of all task AUC scores)")
        print(f"   AUC Mean: {ss_auc_mean:.4f} (average AUC per task)")
        print(f"   Top-10 Sum: {ss_top10_sum:.4f} (sum of all task top-10 scores)")
        print(f"   Best Score Sum: {ss_best_sum:.4f} (sum of best scores from each task)")
        print(f"   Best Score Max: {ss_best_max:.4f} (highest single molecule score)")
        print(f"   Total Runs: {ss_runs}")
        print(f"   Total Molecules: {ss_molecules}")

        print(f"\nITERATIVE PIPELINE SUMMARY:")
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

        print(f"\nDIFFERENCES (Iterative - Single-Shot):")
        print(f"   AUC Sum Difference: {auc_sum_diff:+.4f}")
        print(f"   AUC Mean Difference: {auc_mean_diff:+.4f}")
        print(f"   Top-10 Sum Difference: {top10_sum_diff:+.4f}")
        print(f"   Best Score Sum Difference: {best_sum_diff:+.4f}")

        # Winners
        auc_sum_winner = "Iterative" if auc_sum_diff > 0 else "Single-Shot" if auc_sum_diff < 0 else "Tie"
        auc_mean_winner = "Iterative" if auc_mean_diff > 0 else "Single-Shot" if auc_mean_diff < 0 else "Tie"
        top10_winner = "Iterative" if top10_sum_diff > 0 else "Single-Shot" if top10_sum_diff < 0 else "Tie"
        best_winner = "Iterative" if best_sum_diff > 0 else "Single-Shot" if best_sum_diff < 0 else "Tie"

        print(f"\nWINNERS:")
        print(f"   AUC Sum Winner: {auc_sum_winner}")
        print(f"   AUC Mean Winner: {auc_mean_winner}")
        print(f"   Top-10 Sum Winner: {top10_winner}")
        print(f"   Best Score Sum Winner: {best_winner}")

    def run_complete_evaluation(self):
        """Run the complete oracle evaluation pipeline"""
        print("Starting Complete Oracle Evaluation with TDC")
        print("=" * 80)

        # Extract experiment results
        experiment_results = self.extract_experiment_results()

        if not experiment_results:
            print("No experiment results found!")
            return None

        print(f"Found results for {len(experiment_results)} queries")

        # Evaluate each query
        all_evaluations = {}
        successful_evaluations = 0

        for query_name, query_results in experiment_results.items():
            print(f"\nProcessing query: {query_name}")
            evaluation = self.evaluate_query_results(query_name, query_results)
            all_evaluations[query_name] = evaluation

            if evaluation is not None:
                successful_evaluations += 1

        print(f"\nSuccessfully evaluated {successful_evaluations}/{len(experiment_results)} queries")

        # Create clean visualizations (separate neat plots)
        df, auc_df = self.create_clean_visualizations(all_evaluations)

        # Create detailed top performers analysis
        self.create_top_performers_analysis(all_evaluations)

        # Create LaTeX tables (includes task-wise comparison)
        ss_df, it_df = self.create_latex_tables(all_evaluations)

        # Save comprehensive results
        results_file = self.results_dir / "complete_tdc_oracle_evaluation.json"
        with open(results_file, 'w') as f:
            json.dump(all_evaluations, f, indent=2, default=str)

        print(f"\nComplete evaluation finished!")
        print(f"Evaluated {successful_evaluations} queries with TDC oracles")
        print(f"Clean visualizations saved to: {self.results_dir / 'visualizations'}")
        print(f"   • AUC performance with error bars")
        print(f"   • Query performance heatmap")
        print(f"   • Score distributions")
        print(f"   • Pipeline efficiency analysis")
        print(f"LaTeX tables saved to: {self.results_dir / 'tables'}")
        print(f"Task-wise comparison table created")
        print(f"Detailed results saved to: {results_file}")

        return all_evaluations, df, auc_df, ss_df, it_df


def main():
    """Main execution function"""
    evaluator = ComprehensiveOracleEvaluator(results_dir="results/Gemini_2.0_Flash_Temp_0.9_Results")

    try:
        results = evaluator.run_complete_evaluation()

        if results:
            all_evaluations, df, auc_df, ss_df, it_df = results

            print("\n" + "=" * 80)
            print("TDC ORACLE EVALUATION COMPLETE!")
            print("=" * 80)

            return results

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()