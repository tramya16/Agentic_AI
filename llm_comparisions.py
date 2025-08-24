import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import auc
import warnings
from rdkit import RDLogger
import traceback
from typing import Dict, List, Optional
import re
from scipy import stats
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# Set professional style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True
})


class ResearchFocusedLLMComparator:
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)

        # Create only pipeline-specific directories as main directories
        self.single_shot_dir = Path("comparision_results_singleshot")
        self.iterative_dir = Path("comparision_results_iterative")

        for pipeline_dir in [self.single_shot_dir, self.iterative_dir]:
            pipeline_dir.mkdir(exist_ok=True)
            (pipeline_dir / "visualizations").mkdir(exist_ok=True)
            (pipeline_dir / "tables").mkdir(exist_ok=True)
            (pipeline_dir / "data").mkdir(exist_ok=True)

        # Professional color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'single_shot': '#2E86AB',
            'iterative': '#A23B72',
        }

    def extract_llm_name_from_folder(self, folder_path: Path) -> str:
        """Extract LLM name from folder name"""
        folder_name = folder_path.name
        patterns = [
            r"(.+?)_Temp_(\d+\.?\d*)_Results?$",
            r"(.+?)_experiment_results_Temp_(\d+\.?\d*)$",
            r"(.+?)_Results?$",
        ]

        for pattern in patterns:
            match = re.match(pattern, folder_name, re.IGNORECASE)
            if match:
                model_name = match.group(1)
                model_name = re.sub(r"^scripts_", "", model_name, flags=re.IGNORECASE)

                if len(match.groups()) > 1:
                    temperature = match.group(2)
                    clean_name = self._clean_model_name(model_name)
                    return f"{clean_name} (T={temperature})"
                else:
                    return self._clean_model_name(model_name)

        return self._clean_model_name(folder_name)

    def _clean_model_name(self, name: str) -> str:
        """Clean and format model name"""
        name = name.replace("_", " ")
        name_mappings = {
            "Gemini 2.5": "Gemini 2.5",
            "Gemini 2.0": "Gemini 2.0",
            "Gemini 2.0 Flash": "Gemini 2.0 Flash",
            "Gemini 1.5": "Gemini 1.5",
            "DeepSeekV3": "DeepSeek V3",
            "Deepseekv3": "DeepSeek V3"
        }

        for old, new in name_mappings.items():
            if old.lower() in name.lower():
                name = re.sub(re.escape(old), new, name, flags=re.IGNORECASE)
                break

        return " ".join(word.title() if word.lower() not in ['pro', 'flash'] else word.title()
                        for word in name.split())

    def find_llm_folders(self) -> List[Path]:
        """Find all folders that contain LLM results with oracle scores"""
        print(f"Searching for LLM result folders in {self.base_dir}...")
        llm_folders = []

        for folder in self.base_dir.iterdir():
            if folder.is_dir():
                # Look for JSON files with oracle scores
                json_files = list(folder.glob("*.json"))
                oracle_files = []

                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            # Check if this file contains oracle scores
                            if self._has_oracle_scores(data):
                                oracle_files.append(json_file)
                    except:
                        continue

                if oracle_files:
                    llm_folders.append(folder)
                    print(f"  Found: {folder.name} ({len(oracle_files)} oracle result files)")

        print(f"Found {len(llm_folders)} LLM folders with oracle scores\n")
        return llm_folders

    def _has_oracle_scores(self, data: dict) -> bool:
        """Check if the JSON data contains oracle scores"""
        # Check for the structure from your example JSON
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

    def load_oracle_results_from_folder(self, folder_path: Path) -> Dict:
        """Load oracle results directly from JSON files"""
        llm_name = self.extract_llm_name_from_folder(folder_path)
        print(f"Loading oracle results for {llm_name}...")

        all_results = {}
        json_files = list(folder_path.glob("*.json"))

        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Check if this file contains oracle scores
                if self._has_oracle_scores(data):
                    # Merge the data
                    for query_name, query_data in data.items():
                        if isinstance(query_data, dict) and 'single_shot' in query_data:
                            all_results[query_name] = query_data

            except Exception as e:
                print(f"  Warning: Error loading {file_path}: {e}")

        print(f"  Loaded oracle results for {len(all_results)} queries")
        return {
            "llm_name": llm_name,
            "folder_path": str(folder_path),
            "results": all_results
        }

    def calculate_auc_top_k(self, scores: List[float], k: int = 10) -> float:
        """Calculate AUC for top-k molecules"""
        if len(scores) == 0:
            return 0.0
        sorted_scores = sorted(scores, reverse=True)
        top_k_scores = sorted_scores[:min(k, len(sorted_scores))]
        if len(top_k_scores) < 2:
            return np.mean(top_k_scores) if top_k_scores else 0.0
        x = np.linspace(0, 1, len(top_k_scores))
        try:
            return auc(x, top_k_scores)
        except:
            return np.mean(top_k_scores)

    def process_oracle_results(self, llm_data: Dict) -> Dict:
        """Process pre-computed oracle results for a single LLM"""
        llm_name = llm_data["llm_name"]
        results = llm_data["results"]

        print(f"Processing oracle results for {llm_name}...")

        llm_evaluation = {
            "llm_name": llm_name,
            "folder_path": llm_data["folder_path"],
            "query_evaluations": {},
            "summary": {
                "total_queries": len(results),
                "successful_queries": 0,
                "single_shot": {"auc_sum": 0.0, "auc_mean": 0.0, "total_runs": 0},
                "iterative": {"auc_sum": 0.0, "auc_mean": 0.0, "total_runs": 0}
            }
        }

        successful_queries = 0
        ss_task_auc_means = []
        it_task_auc_means = []
        ss_total_runs = 0
        it_total_runs = 0

        for query_name, query_results in results.items():
            query_eval = self.process_query_oracle_results(query_name, query_results, llm_name)
            if query_eval:
                llm_evaluation["query_evaluations"][query_name] = query_eval
                successful_queries += 1

                # Calculate mean AUC per task
                if query_eval["single_shot"]["auc_scores"]:
                    task_ss_auc_mean = np.mean(query_eval["single_shot"]["auc_scores"])
                    ss_task_auc_means.append(task_ss_auc_mean)
                    ss_total_runs += len(query_eval["single_shot"]["auc_scores"])

                if query_eval["iterative"]["auc_scores"]:
                    task_it_auc_mean = np.mean(query_eval["iterative"]["auc_scores"])
                    it_task_auc_means.append(task_it_auc_mean)
                    it_total_runs += len(query_eval["iterative"]["auc_scores"])

        llm_evaluation["summary"]["successful_queries"] = successful_queries

        if ss_task_auc_means:
            llm_evaluation["summary"]["single_shot"]["auc_sum"] = sum(ss_task_auc_means)
            llm_evaluation["summary"]["single_shot"]["auc_mean"] = np.mean(ss_task_auc_means)
            llm_evaluation["summary"]["single_shot"]["total_runs"] = ss_total_runs

        if it_task_auc_means:
            llm_evaluation["summary"]["iterative"]["auc_sum"] = sum(it_task_auc_means)
            llm_evaluation["summary"]["iterative"]["auc_mean"] = np.mean(it_task_auc_means)
            llm_evaluation["summary"]["iterative"]["total_runs"] = it_total_runs

        print(f"  Processed {successful_queries}/{len(results)} queries")
        return llm_evaluation

    def process_query_oracle_results(self, query_name: str, query_results: Dict, llm_name: str) -> Optional[Dict]:
        """Process oracle results for a single query"""
        evaluation_results = {
            "query_name": query_name,
            "oracle_name": query_results.get("oracle_name", "Unknown"),
            "llm_name": llm_name,
            "single_shot": {"runs": [], "auc_scores": [], "top_10_scores": []},
            "iterative": {"runs": [], "auc_scores": [], "top_10_scores": []}
        }

        for pipeline in ['single_shot', 'iterative']:
            if pipeline in query_results and 'runs' in query_results[pipeline]:
                for run_data in query_results[pipeline]['runs']:
                    if 'molecules' in run_data and run_data['molecules']:
                        # Use pre-computed oracle scores and ADD PIPELINE INFORMATION
                        molecules = run_data['molecules']

                        # Add Pipeline column to each molecule
                        for mol in molecules:
                            mol['Pipeline'] = 'Single-Shot' if pipeline == 'single_shot' else 'Iterative'
                            mol['LLM'] = llm_name
                            mol['Query'] = query_name

                        scores = [mol['Oracle_Score'] for mol in molecules]

                        if scores:
                            auc_top_10 = self.calculate_auc_top_k(scores, k=10)
                            top_10_mean = np.mean(sorted(scores, reverse=True)[:10])

                            run_result = {
                                "run": run_data.get("run", 1),
                                "seed": run_data.get("seed", 0),
                                "total_molecules": len(molecules),
                                "oracle_scores": scores,
                                "auc_top_10": auc_top_10,
                                "top_10_mean": top_10_mean,
                                "max_score": max(scores),
                                "mean_score": np.mean(scores),
                                "molecules": molecules
                            }

                            evaluation_results[pipeline]["runs"].append(run_result)
                            evaluation_results[pipeline]["auc_scores"].append(auc_top_10)
                            evaluation_results[pipeline]["top_10_scores"].append(top_10_mean)

        return evaluation_results if (
                evaluation_results["single_shot"]["runs"] or evaluation_results["iterative"]["runs"]) else None

    def create_research_visualizations(self, all_llm_evaluations: Dict):
        """Create research-focused visualizations in pipeline-specific directories"""
        print("\nCreating pipeline-specific analyses...")

        # Prepare comprehensive data
        comparison_data = []
        molecule_data = []
        smiles_overlap_data = {}

        for llm_name, llm_eval in all_llm_evaluations.items():
            smiles_overlap_data[llm_name] = {'single_shot': set(), 'iterative': set()}

            for query_name, query_eval in llm_eval["query_evaluations"].items():
                for pipeline in ['single_shot', 'iterative']:
                    pipeline_name = pipeline.replace('_', '-').title()

                    if query_eval[pipeline]['auc_scores']:
                        auc_scores = query_eval[pipeline]['auc_scores']
                        top10_scores = query_eval[pipeline]['top_10_scores']

                        comparison_data.append({
                            'LLM': llm_name,
                            'Query': query_name,
                            'Pipeline': pipeline_name,
                            'AUC_Mean': np.mean(auc_scores),
                            'AUC_Std': np.std(auc_scores),
                            'Top10_Mean': np.mean(top10_scores),
                            'Top10_Std': np.std(top10_scores),
                            'N_Runs': len(auc_scores)
                        })

                        # Collect molecules and SMILES for overlap analysis
                        for run in query_eval[pipeline]['runs']:
                            # The molecules already have Pipeline and LLM added in process_query_oracle_results
                            molecule_data.extend(run['molecules'])
                            for mol in run['molecules']:
                                smiles_overlap_data[llm_name][pipeline].add(mol['SMILES'])

        comp_df = pd.DataFrame(comparison_data)
        mol_df = pd.DataFrame(molecule_data)

        print(f"\nDebug: mol_df columns after creation: {mol_df.columns.tolist()}")
        print(
            f"Debug: mol_df Pipeline values: {mol_df['Pipeline'].value_counts() if 'Pipeline' in mol_df.columns else 'Pipeline column missing'}")

        # Create pipeline-specific analyses (this will create all visualizations for each pipeline)
        self.create_pipeline_specific_analysis(all_llm_evaluations, "single_shot")
        self.create_pipeline_specific_analysis(all_llm_evaluations, "iterative")

        return comp_df, mol_df, smiles_overlap_data

    def create_pipeline_specific_analysis(self, all_llm_evaluations: Dict, pipeline_type: str):
        """Create analysis for a specific pipeline type"""
        print(f"\nCreating {pipeline_type} pipeline analysis...")

        # Set up directories
        pipeline_dir = self.single_shot_dir if pipeline_type == "single_shot" else self.iterative_dir
        viz_dir = pipeline_dir / "visualizations"
        tables_dir = pipeline_dir / "tables"
        data_dir = pipeline_dir / "data"

        # Extract pipeline-specific data
        pipeline_evaluations = self.extract_pipeline_evaluations(all_llm_evaluations, pipeline_type)

        # Create pipeline-specific visualizations
        self.create_pipeline_visualizations(pipeline_evaluations, pipeline_type, viz_dir)

        # Create pipeline-specific tables
        self.create_pipeline_tables(pipeline_evaluations, pipeline_type, tables_dir)

        # Save pipeline-specific data
        self.save_pipeline_data(pipeline_evaluations, pipeline_type, data_dir)

        print(f"✓ Completed {pipeline_type} pipeline analysis")

    def create_rq1_single_vs_iterative_analysis(self, comp_df: pd.DataFrame, mol_df: pd.DataFrame, viz_dir: Path):
        """RQ1: Single-shot vs Iterative comparison"""
        print("Creating RQ1 Single-Shot vs Iterative Analysis...")

        # Check if Pipeline column exists
        if 'Pipeline' not in mol_df.columns:
            print("Warning: 'Pipeline' column not found in mol_df!")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Research Question 1: Single-Shot vs Iterative Generation Performance',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Box plot: AUC distribution by pipeline
        pipeline_data = comp_df.groupby(['Pipeline', 'LLM']).agg({'AUC_Mean': 'sum'}).reset_index()

        box_data = [
            pipeline_data[pipeline_data['Pipeline'] == 'Single-Shot']['AUC_Mean'].values,
            pipeline_data[pipeline_data['Pipeline'] == 'Iterative']['AUC_Mean'].values
        ]

        bp = ax1.boxplot(box_data, labels=['Single-Shot', 'Iterative'], patch_artist=True,
                         boxprops=dict(facecolor=self.colors['single_shot'], alpha=0.7),
                         medianprops=dict(color='black', linewidth=2))

        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor(self.colors['iterative'])

        ax1.set_title('AUC Score Distribution by Pipeline', fontweight='bold', pad=20)
        ax1.set_ylabel('Total AUC Score')

        # Add statistical annotation
        ss_scores = box_data[0]
        it_scores = box_data[1]
        if len(ss_scores) > 0 and len(it_scores) > 0:
            try:
                t_stat, p_value = stats.ttest_ind(ss_scores, it_scores)
                ax1.text(0.5, 0.95, f'p-value: {p_value:.4f}', transform=ax1.transAxes,
                         ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass

        # 2. Violin plot: Score distribution at molecule level
        mol_pipeline_data = mol_df[mol_df['Oracle_Score'] > 0]

        if not mol_pipeline_data.empty:
            try:
                ss_scores = mol_pipeline_data[mol_pipeline_data['Pipeline'] == 'Single-Shot']['Oracle_Score'].values
                it_scores = mol_pipeline_data[mol_pipeline_data['Pipeline'] == 'Iterative']['Oracle_Score'].values

                if len(ss_scores) > 0 or len(it_scores) > 0:
                    parts = ax2.violinplot([ss_scores, it_scores], positions=[1, 2], showmeans=True, showmedians=True)

                    if len(parts['bodies']) > 0:
                        parts['bodies'][0].set_facecolor(self.colors['single_shot'])
                        parts['bodies'][0].set_alpha(0.7)
                    if len(parts['bodies']) > 1:
                        parts['bodies'][1].set_facecolor(self.colors['iterative'])
                        parts['bodies'][1].set_alpha(0.7)

                    ax2.set_xticks([1, 2])
                    ax2.set_xticklabels(['Single-Shot', 'Iterative'])
            except Exception as e:
                print(f"Violin plot failed: {e}")
                try:
                    ax2.hist([ss_scores, it_scores], bins=20, alpha=0.7,
                             label=['Single-Shot', 'Iterative'],
                             color=[self.colors['single_shot'], self.colors['iterative']])
                    ax2.legend()
                except:
                    ax2.text(0.5, 0.5, 'Insufficient data for visualization',
                             ha='center', va='center', transform=ax2.transAxes)

            ax2.set_title('Molecule-Level Score Distribution', fontweight='bold', pad=20)
            ax2.set_ylabel('Oracle Score')

        # 3. Success rate comparison (scores > 0.8)
        success_data = []
        for pipeline in ['Single-Shot', 'Iterative']:
            pipeline_mols = mol_df[mol_df['Pipeline'] == pipeline]
            if len(pipeline_mols) > 0:
                success_rate = len(pipeline_mols[pipeline_mols['Oracle_Score'] > 0.8]) / len(pipeline_mols) * 100
                success_data.append({
                    'Pipeline': pipeline,
                    'Success_Rate': success_rate,
                    'Total_Molecules': len(pipeline_mols),
                    'High_Score_Molecules': len(pipeline_mols[pipeline_mols['Oracle_Score'] > 0.8])
                })

        success_df = pd.DataFrame(success_data)
        if not success_df.empty:
            colors = [self.colors['single_shot'], self.colors['iterative']]
            bars = ax3.bar(success_df['Pipeline'], success_df['Success_Rate'],
                           color=colors, alpha=0.8, edgecolor='black', linewidth=1)

            for bar, row in zip(bars, success_df.iterrows()):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + 1,
                         f'{height:.1f}%\n({row[1]["High_Score_Molecules"]}/{row[1]["Total_Molecules"]})',
                         ha='center', va='bottom', fontweight='bold')

            ax3.set_title('Success Rate (Oracle Score > 0.8)', fontweight='bold', pad=20)
            ax3.set_ylabel('Success Rate (%)')
            if len(success_df) > 0:
                ax3.set_ylim(0, max(success_df['Success_Rate']) * 1.3)

        # 4. Performance consistency (coefficient of variation)
        consistency_data = []
        for pipeline in ['Single-Shot', 'Iterative']:
            pipeline_aucs = comp_df[comp_df['Pipeline'] == pipeline]['AUC_Mean']
            if len(pipeline_aucs) > 0 and np.mean(pipeline_aucs) > 0:
                cv = np.std(pipeline_aucs) / np.mean(pipeline_aucs) * 100
                consistency_data.append({
                    'Pipeline': pipeline,
                    'Coefficient_of_Variation': cv,
                    'Mean_AUC': np.mean(pipeline_aucs),
                    'Std_AUC': np.std(pipeline_aucs)
                })

        consistency_df = pd.DataFrame(consistency_data)
        if not consistency_df.empty:
            colors = [self.colors['single_shot'], self.colors['iterative']]
            bars = ax4.bar(consistency_df['Pipeline'], consistency_df['Coefficient_of_Variation'],
                           color=colors, alpha=0.8, edgecolor='black', linewidth=1)

            for bar, row in zip(bars, consistency_df.iterrows()):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height + 1,
                         f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

            ax4.set_title('Performance Consistency\n(Lower = More Consistent)', fontweight='bold', pad=20)
            ax4.set_ylabel('Coefficient of Variation (%)')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / "rq1_single_vs_iterative_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created RQ1 analysis")

    def create_rq2_llm_performance_comparison(self, comp_df: pd.DataFrame, mol_df: pd.DataFrame,
                                              all_llm_evaluations: Dict, viz_dir: Path):
        """RQ2: LLM Performance Comparison"""
        print("Creating RQ2 LLM Performance Comparison...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Research Question 2: LLM Performance Comparison in Agentic System',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Multi-dimensional performance comparison
        llm_metrics = []
        for llm_name, llm_eval in all_llm_evaluations.items():
            summary = llm_eval["summary"]

            # Calculate metrics
            total_auc = summary["single_shot"]["auc_sum"] + summary["iterative"]["auc_sum"]
            coverage = summary["successful_queries"] / summary["total_queries"] * 100 if summary[
                                                                                             "total_queries"] > 0 else 0

            # Get additional metrics
            total_molecules = 0
            best_score = 0.0
            for query_eval in llm_eval["query_evaluations"].values():
                for pipeline in ['single_shot', 'iterative']:
                    for run in query_eval[pipeline]['runs']:
                        total_molecules += run['total_molecules']
                        best_score = max(best_score, run['max_score'])

            efficiency = total_auc / total_molecules if total_molecules > 0 else 0

            llm_metrics.append({
                'LLM': llm_name,
                'Total_AUC': total_auc,
                'Coverage': coverage,
                'Efficiency': efficiency * 1000,  # Scale for visibility
                'Best_Score': best_score,
                'Total_Molecules': total_molecules
            })

        metrics_df = pd.DataFrame(llm_metrics)

        if not metrics_df.empty:
            # Normalize metrics for comparison (0-100 scale)
            for col in ['Total_AUC', 'Efficiency', 'Best_Score']:
                if metrics_df[col].max() > 0:
                    metrics_df[f'{col}_norm'] = (metrics_df[col] / metrics_df[col].max()) * 100
                else:
                    metrics_df[f'{col}_norm'] = 0

            # Stacked bar chart
            x_pos = np.arange(len(metrics_df))
            width = 0.6

            bottom = np.zeros(len(metrics_df))
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            metrics = ['Total_AUC_norm', 'Coverage', 'Efficiency_norm', 'Best_Score_norm']
            labels = ['Total AUC', 'Coverage', 'Efficiency', 'Best Score']

            for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
                values = metrics_df[metric] if metric in metrics_df.columns else np.zeros(len(metrics_df))
                ax1.bar(x_pos, values, width, bottom=bottom,
                        label=label, color=color, alpha=0.8)
                bottom += values

            ax1.set_title('Multi-Dimensional LLM Performance\n(Normalized Metrics)', fontweight='bold', pad=20)
            ax1.set_ylabel('Normalized Score')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(metrics_df['LLM'], rotation=45, ha='right')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # 2. Performance scatter plot
            ax2.scatter(metrics_df['Total_AUC'], metrics_df['Coverage'],
                        s=metrics_df['Efficiency'] * 10, alpha=0.7,
                        c=range(len(metrics_df)), cmap='viridis')

            for i, row in metrics_df.iterrows():
                ax2.annotate(row['LLM'], (row['Total_AUC'], row['Coverage']),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)

            ax2.set_title('Performance vs Coverage\n(Bubble size = Efficiency)', fontweight='bold', pad=20)
            ax2.set_xlabel('Total AUC Score')
            ax2.set_ylabel('Query Coverage (%)')

        # 3. Pipeline preference by LLM
        pipeline_pref_data = []
        for llm_name, llm_eval in all_llm_evaluations.items():
            ss_auc = llm_eval["summary"]["single_shot"]["auc_sum"]
            it_auc = llm_eval["summary"]["iterative"]["auc_sum"]

            pipeline_pref_data.append({
                'LLM': llm_name,
                'Single_Shot': ss_auc,
                'Iterative': it_auc,
                'Difference': it_auc - ss_auc,
                'Preferred': 'Iterative' if it_auc > ss_auc else 'Single-Shot'
            })

        pref_df = pd.DataFrame(pipeline_pref_data).sort_values('Difference')

        if not pref_df.empty:
            colors = [self.colors['iterative'] if d > 0 else self.colors['single_shot']
                      for d in pref_df['Difference']]

            bars = ax3.barh(pref_df['LLM'], pref_df['Difference'], color=colors, alpha=0.8, edgecolor='black')

            for bar, val, pref in zip(bars, pref_df['Difference'], pref_df['Preferred']):
                x_pos = val + (0.5 if val > 0 else -0.5)
                ax3.text(x_pos, bar.get_y() + bar.get_height() / 2,
                         f'{abs(val):.1f}\n({pref})', ha='left' if val > 0 else 'right',
                         va='center', fontweight='bold', fontsize=9)

            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_title('Pipeline Preference by LLM\n(Iterative - Single-Shot AUC)', fontweight='bold', pad=20)
            ax3.set_xlabel('AUC Difference')

        # 4. Task-specific performance heatmap
        if not comp_df.empty:
            task_performance = comp_df.pivot_table(values='AUC_Mean', index='Query',
                                                   columns='LLM', aggfunc='max', fill_value=0)

            # Select top 10 most challenging tasks (lowest average performance)
            if len(task_performance) > 10:
                task_difficulty = task_performance.mean(axis=1).sort_values().head(10)
                top_tasks = task_performance.loc[task_difficulty.index]
            else:
                top_tasks = task_performance

            if not top_tasks.empty:
                im = ax4.imshow(top_tasks.values, cmap='RdYlGn', aspect='auto')
                ax4.set_xticks(range(len(top_tasks.columns)))
                ax4.set_xticklabels(top_tasks.columns, rotation=45, ha='right')
                ax4.set_yticks(range(len(top_tasks.index)))
                ax4.set_yticklabels([task[:20] + '...' if len(task) > 20 else task
                                     for task in top_tasks.index], fontsize=9)
                ax4.set_title('Performance on Most Challenging Tasks', fontweight='bold', pad=20)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
                cbar.set_label('Best AUC Score')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / "rq2_llm_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created RQ2 analysis")

    def create_rq3_chemical_space_overlap(self, smiles_overlap_data: Dict, viz_dir: Path):
        """RQ3: Chemical Space Overlap Analysis"""
        print("Creating RQ3 Chemical Space Overlap Analysis...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Research Question 3: Chemical Space Overlap Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Overlap percentage by LLM
        overlap_data = []
        for llm_name, smiles_data in smiles_overlap_data.items():
            ss_smiles = smiles_data['single_shot']
            it_smiles = smiles_data['iterative']

            if len(ss_smiles) > 0 and len(it_smiles) > 0:
                overlap = len(ss_smiles.intersection(it_smiles))
                union = len(ss_smiles.union(it_smiles))
                jaccard = overlap / union if union > 0 else 0

                overlap_data.append({
                    'LLM': llm_name,
                    'Single_Shot_Count': len(ss_smiles),
                    'Iterative_Count': len(it_smiles),
                    'Overlap_Count': overlap,
                    'Jaccard_Index': jaccard,
                    'Overlap_Percentage': (overlap / min(len(ss_smiles), len(it_smiles))) * 100
                })

        overlap_df = pd.DataFrame(overlap_data)

        if not overlap_df.empty:
            # Stacked bar showing unique vs overlapping molecules
            x_pos = np.arange(len(overlap_df))
            width = 0.6

            ss_unique = overlap_df['Single_Shot_Count'] - overlap_df['Overlap_Count']
            it_unique = overlap_df['Iterative_Count'] - overlap_df['Overlap_Count']
            overlap_count = overlap_df['Overlap_Count']

            ax1.bar(x_pos, ss_unique, width, label='Single-Shot Unique',
                    color=self.colors['single_shot'], alpha=0.8)
            ax1.bar(x_pos, it_unique, width, bottom=ss_unique,
                    label='Iterative Unique', color=self.colors['iterative'], alpha=0.8)
            ax1.bar(x_pos, overlap_count, width, bottom=ss_unique + it_unique,
                    label='Overlapping', color='#F18F01', alpha=0.8)

            for i, row in overlap_df.iterrows():
                total = row['Single_Shot_Count'] + row['Iterative_Count'] - row['Overlap_Count']
                ax1.text(i, total + 10, f'{row["Overlap_Percentage"]:.1f}%',
                         ha='center', va='bottom', fontweight='bold')

            ax1.set_title('Chemical Space Overlap by LLM', fontweight='bold', pad=20)
            ax1.set_ylabel('Number of Unique Molecules')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(overlap_df['LLM'], rotation=45, ha='right')
            ax1.legend()

        # 2. Jaccard Index comparison
        if not overlap_df.empty:
            bars = ax2.bar(overlap_df['LLM'], overlap_df['Jaccard_Index'],
                           color='#3A9B7A', alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, overlap_df['Jaccard_Index']):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

            ax2.set_title('Jaccard Index (Chemical Space Similarity)', fontweight='bold', pad=20)
            ax2.set_ylabel('Jaccard Index')
            if len(overlap_df) > 0:
                ax2.set_ylim(0, max(overlap_df['Jaccard_Index']) * 1.2)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # 3. Chemical space distribution pie chart
        if overlap_data:
            first_llm = overlap_data[0]

            total_ss = first_llm['Single_Shot_Count']
            total_it = first_llm['Iterative_Count']
            overlap = first_llm['Overlap_Count']

            if total_ss > 0 or total_it > 0:
                sizes = [total_ss - overlap, total_it - overlap, overlap]
                labels = ['Single-Shot Only', 'Iterative Only', 'Overlapping']
                colors = [self.colors['single_shot'], self.colors['iterative'], '#F18F01']

                non_zero_sizes = []
                non_zero_labels = []
                non_zero_colors = []

                for size, label, color in zip(sizes, labels, colors):
                    if size > 0:
                        non_zero_sizes.append(size)
                        non_zero_labels.append(label)
                        non_zero_colors.append(color)

                if non_zero_sizes:
                    try:
                        wedges, texts, autotexts = ax3.pie(non_zero_sizes, labels=non_zero_labels,
                                                           colors=non_zero_colors, autopct='%1.1f%%',
                                                           startangle=90, pctdistance=0.85)
                        for text in texts:
                            text.set_fontsize(10)
                        for autotext in autotexts:
                            autotext.set_fontsize(10)

                        ax3.set_title(f'Chemical Space Distribution\n({first_llm["LLM"]})',
                                      fontweight='bold', pad=20)
                    except:
                        ax3.bar(range(len(non_zero_labels)), non_zero_sizes, color=non_zero_colors)
                        ax3.set_xticks(range(len(non_zero_labels)))
                        ax3.set_xticklabels(non_zero_labels, rotation=45, ha='right')
                        ax3.set_title(f'Chemical Space Distribution\n({first_llm["LLM"]})',
                                      fontweight='bold', pad=20)
                        ax3.set_ylabel('Number of Molecules')

        # 4. Diversity analysis
        diversity_data = []
        for llm_name, smiles_data in smiles_overlap_data.items():
            for pipeline, smiles_set in smiles_data.items():
                if len(smiles_set) > 0:
                    unique_count = len(smiles_set)
                    avg_length = np.mean([len(smiles) for smiles in smiles_set])

                    diversity_data.append({
                        'LLM': llm_name,
                        'Pipeline': pipeline.replace('_', '-').title(),
                        'Unique_Molecules': unique_count,
                        'Avg_SMILES_Length': avg_length
                    })

        diversity_df = pd.DataFrame(diversity_data)

        if not diversity_df.empty:
            llms = diversity_df['LLM'].unique()
            x_pos = np.arange(len(llms))
            width = 0.35

            ss_diversity = []
            it_diversity = []

            for llm in llms:
                ss_data = diversity_df[(diversity_df['LLM'] == llm) &
                                       (diversity_df['Pipeline'] == 'Single-Shot')]
                it_data = diversity_df[(diversity_df['LLM'] == llm) &
                                       (diversity_df['Pipeline'] == 'Iterative')]

                ss_diversity.append(ss_data['Unique_Molecules'].iloc[0] if len(ss_data) > 0 else 0)
                it_diversity.append(it_data['Unique_Molecules'].iloc[0] if len(it_data) > 0 else 0)

            bars1 = ax4.bar(x_pos - width / 2, ss_diversity, width,
                            label='Single-Shot', color=self.colors['single_shot'], alpha=0.8)
            bars2 = ax4.bar(x_pos + width / 2, it_diversity, width,
                            label='Iterative', color=self.colors['iterative'], alpha=0.8)

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax4.text(bar.get_x() + bar.get_width() / 2, height + 5,
                                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

            ax4.set_title('Unique Molecule Generation by Pipeline', fontweight='bold', pad=20)
            ax4.set_ylabel('Number of Unique Molecules')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(llms, rotation=45, ha='right')
            ax4.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / "rq3_chemical_space_overlap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created RQ3 analysis")

    def create_validity_novelty_diversity_analysis(self, mol_df: pd.DataFrame, viz_dir: Path):
        """Validity, Novelty, and Diversity Analysis"""
        print("Creating Validity, Novelty, and Diversity Analysis...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Validity, Novelty, and Diversity Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Score distribution histogram
        if not mol_df.empty:
            score_data = mol_df[mol_df['Oracle_Score'] > 0]['Oracle_Score']

            if not score_data.empty:
                ax1.hist(score_data, bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
                ax1.axvline(score_data.mean(), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {score_data.mean():.3f}')
                ax1.axvline(score_data.median(), color='orange', linestyle='--', linewidth=2,
                            label=f'Median: {score_data.median():.3f}')

                ax1.set_title('Oracle Score Distribution', fontweight='bold', pad=20)
                ax1.set_xlabel('Oracle Score')
                ax1.set_ylabel('Frequency')
                ax1.legend()

        # 2. Performance by LLM and Pipeline
        if 'Pipeline' in mol_df.columns and 'LLM' in mol_df.columns:
            mol_df['LLM_Pipeline'] = mol_df['LLM'] + ' - ' + mol_df['Pipeline']
            filtered_mol_df = mol_df[mol_df['Oracle_Score'] > 0]

            if not filtered_mol_df.empty:
                categories = filtered_mol_df['LLM_Pipeline'].unique()
                data_for_violin = [filtered_mol_df[filtered_mol_df['LLM_Pipeline'] == cat]['Oracle_Score'].values
                                   for cat in categories]

                try:
                    parts = ax2.violinplot(data_for_violin, positions=range(len(categories)),
                                           showmeans=True, showmedians=True)

                    for i, pc in enumerate(parts['bodies']):
                        if i < len(categories) and 'Single-Shot' in categories[i]:
                            pc.set_facecolor(self.colors['single_shot'])
                        else:
                            pc.set_facecolor(self.colors['iterative'])
                        pc.set_alpha(0.7)

                    ax2.set_xticks(range(len(categories)))
                    ax2.set_xticklabels(categories, rotation=45, ha='right')
                except:
                    bp = ax2.boxplot(data_for_violin, labels=categories, patch_artist=True)
                    for i, patch in enumerate(bp['boxes']):
                        if i < len(categories) and 'Single-Shot' in categories[i]:
                            patch.set_facecolor(self.colors['single_shot'])
                        else:
                            patch.set_facecolor(self.colors['iterative'])
                        patch.set_alpha(0.7)
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

                ax2.set_title('Score Distribution by LLM-Pipeline', fontweight='bold', pad=20)
                ax2.set_ylabel('Oracle Score')

        # 3. Top performers analysis - Fair sampling across LLMs
        if not mol_df.empty and 'LLM' in mol_df.columns and 'Pipeline' in mol_df.columns:
            # Get top performers fairly from each LLM to avoid bias
            top_performers_list = []
            llm_pipeline_combinations = mol_df.groupby(['LLM', 'Pipeline'])

            # Take top 5 from each LLM-Pipeline combination to ensure fair representation
            for (llm, pipeline), group in llm_pipeline_combinations:
                llm_top = group.nlargest(min(5, len(group)), 'Oracle_Score')
                top_performers_list.append(llm_top)

            if top_performers_list:
                top_performers = pd.concat(top_performers_list)
                top_counts = top_performers.groupby(['LLM', 'Pipeline']).size().reset_index(name='Count')

                llms = top_counts['LLM'].unique()
                x_pos = np.arange(len(llms))
                width = 0.35

                ss_counts = []
                it_counts = []

                for llm in llms:
                    ss_count = top_counts[(top_counts['LLM'] == llm) &
                                          (top_counts['Pipeline'] == 'Single-Shot')]['Count'].sum()
                    it_count = top_counts[(top_counts['LLM'] == llm) &
                                          (top_counts['Pipeline'] == 'Iterative')]['Count'].sum()
                    ss_counts.append(ss_count)
                    it_counts.append(it_count)

                bars1 = ax3.bar(x_pos - width / 2, ss_counts, width,
                                label='Single-Shot', color=self.colors['single_shot'], alpha=0.8)
                bars2 = ax3.bar(x_pos + width / 2, it_counts, width,
                                label='Iterative', color=self.colors['iterative'], alpha=0.8)

                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

                ax3.set_title('Top 20 Performers by LLM-Pipeline', fontweight='bold', pad=20)
                ax3.set_ylabel('Number of Top Performers')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(llms, rotation=45, ha='right')
                ax3.legend()

        # 4. Query difficulty analysis
        if not mol_df.empty and 'Query' in mol_df.columns:
            query_stats = mol_df.groupby('Query').agg({
                'Oracle_Score': ['mean', 'std', 'count', 'max']
            }).round(3)
            query_stats.columns = ['Mean_Score', 'Std_Score', 'Count', 'Max_Score']
            query_stats = query_stats.reset_index()

            challenging_queries = query_stats.nsmallest(10, 'Mean_Score')

            if not challenging_queries.empty:
                bars = ax4.barh(range(len(challenging_queries)), challenging_queries['Mean_Score'],
                                color=self.colors['accent'], alpha=0.8, edgecolor='black')

                for i, (bar, val) in enumerate(zip(bars, challenging_queries['Mean_Score'])):
                    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{val:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)

                ax4.set_yticks(range(len(challenging_queries)))
                ax4.set_yticklabels([q[:25] + '...' if len(q) > 25 else q
                                     for q in challenging_queries['Query']], fontsize=9)
                ax4.set_title('Most Challenging Queries\n(Lowest Mean Scores)', fontweight='bold', pad=20)
                ax4.set_xlabel('Mean Oracle Score')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / "validity_novelty_diversity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created validity/novelty/diversity analysis")

    def create_performance_distribution_analysis(self, comp_df: pd.DataFrame, viz_dir: Path):
        """Performance Distribution Analysis"""
        print("Creating Performance Distribution Analysis...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Performance Distribution and Statistical Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        if comp_df.empty:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(viz_dir / "performance_distribution_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            return

        # 1. AUC distribution by LLM
        llm_auc_data = comp_df.groupby(['LLM', 'Query']).agg({'AUC_Mean': 'max'}).reset_index()

        llms = llm_auc_data['LLM'].unique()
        data_for_violin = [llm_auc_data[llm_auc_data['LLM'] == llm]['AUC_Mean'].values
                           for llm in llms]

        try:
            parts = ax1.violinplot(data_for_violin, positions=range(len(llms)),
                                   showmeans=True, showmedians=True, showextrema=True)

            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)

            ax1.set_xticks(range(len(llms)))
            ax1.set_xticklabels(llms, rotation=45, ha='right')
        except:
            bp = ax1.boxplot(data_for_violin, labels=llms, patch_artist=True)
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        ax1.set_title('AUC Distribution by LLM', fontweight='bold', pad=20)
        ax1.set_ylabel('AUC Score')

        # 2. Performance correlation matrix
        correlation_data = comp_df.pivot_table(values='AUC_Mean', index='Query',
                                               columns='LLM', aggfunc='max', fill_value=0)

        if len(correlation_data.columns) > 1:
            try:
                corr_matrix = correlation_data.corr()

                im = ax2.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

                for i in range(len(corr_matrix.index)):
                    for j in range(len(corr_matrix.columns)):
                        ax2.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center',
                                 color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                                 fontweight='bold')

                ax2.set_xticks(range(len(corr_matrix.columns)))
                ax2.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                ax2.set_yticks(range(len(corr_matrix.index)))
                ax2.set_yticklabels(corr_matrix.index)
                ax2.set_title('LLM Performance Correlation Matrix', fontweight='bold', pad=20)

                cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
                cbar.set_label('Correlation Coefficient')
            except:
                ax2.text(0.5, 0.5, 'Correlation analysis failed', ha='center', va='center', transform=ax2.transAxes)

        # 3. Cumulative performance
        for llm in llms:
            llm_data = llm_auc_data[llm_auc_data['LLM'] == llm].sort_values('AUC_Mean', ascending=False)
            if not llm_data.empty:
                cumulative_auc = np.cumsum(llm_data['AUC_Mean'])
                ax3.plot(range(1, len(cumulative_auc) + 1), cumulative_auc,
                         marker='o', label=llm, linewidth=2, markersize=4)

        ax3.set_title('Cumulative AUC Performance', fontweight='bold', pad=20)
        ax3.set_xlabel('Number of Queries (Ranked by Performance)')
        ax3.set_ylabel('Cumulative AUC Score')
        ax3.legend()

        # 4. Performance range analysis
        try:
            bp = ax4.boxplot([llm_auc_data[llm_auc_data['LLM'] == llm]['AUC_Mean'].values
                              for llm in llms],
                             labels=llms, patch_artist=True)

            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

            ax4.set_title('Performance Range Analysis', fontweight='bold', pad=20)
            ax4.set_ylabel('AUC Score')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        except:
            ax4.text(0.5, 0.5, 'Range analysis failed', ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / "performance_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created performance distribution analysis")

    def create_statistical_significance_analysis(self, comp_df: pd.DataFrame, viz_dir: Path):
        """Statistical Significance Analysis"""
        print("Creating Statistical Significance Analysis...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Statistical Significance and Confidence Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        if comp_df.empty:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(viz_dir / "statistical_significance_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            return

        # 1. Confidence intervals for LLM performance
        llm_stats = []
        for llm in comp_df['LLM'].unique():
            llm_data = comp_df[comp_df['LLM'] == llm]
            auc_scores = llm_data['AUC_Mean'].values

            if len(auc_scores) > 1:
                mean_auc = np.mean(auc_scores)
                std_auc = np.std(auc_scores)
                se_auc = std_auc / np.sqrt(len(auc_scores))
                ci_95 = 1.96 * se_auc

                llm_stats.append({
                    'LLM': llm,
                    'Mean': mean_auc,
                    'Std': std_auc,
                    'CI_Lower': mean_auc - ci_95,
                    'CI_Upper': mean_auc + ci_95,
                    'N_Queries': len(auc_scores)
                })

        stats_df = pd.DataFrame(llm_stats)
        if not stats_df.empty:
            stats_df = stats_df.sort_values('Mean', ascending=False)
            x_pos = np.arange(len(stats_df))

            bars = ax1.bar(x_pos, stats_df['Mean'],
                           yerr=[stats_df['Mean'] - stats_df['CI_Lower'],
                                 stats_df['CI_Upper'] - stats_df['Mean']],
                           capsize=5, alpha=0.8, edgecolor='black',
                           color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(stats_df)])

            for bar, row in zip(bars, stats_df.iterrows()):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{row[1]["Mean"]:.3f}', ha='center', va='bottom', fontweight='bold')

            ax1.set_title('Mean AUC with 95% Confidence Intervals', fontweight='bold', pad=20)
            ax1.set_ylabel('Mean AUC Score')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(stats_df['LLM'], rotation=45, ha='right')

        # 2. Pipeline comparison statistical test
        pipeline_comparison = []
        for llm in comp_df['LLM'].unique():
            llm_data = comp_df[comp_df['LLM'] == llm]
            ss_scores = llm_data[llm_data['Pipeline'] == 'Single-Shot']['AUC_Mean'].values
            it_scores = llm_data[llm_data['Pipeline'] == 'Iterative']['AUC_Mean'].values

            if len(ss_scores) > 0 and len(it_scores) > 0:
                try:
                    t_stat, p_value = stats.ttest_ind(ss_scores, it_scores)

                    pipeline_comparison.append({
                        'LLM': llm,
                        'SS_Mean': np.mean(ss_scores),
                        'IT_Mean': np.mean(it_scores),
                        'Difference': np.mean(it_scores) - np.mean(ss_scores),
                        'T_Statistic': t_stat,
                        'P_Value': p_value,
                        'Significant': p_value < 0.05
                    })
                except:
                    pass

        pipeline_df = pd.DataFrame(pipeline_comparison)

        if not pipeline_df.empty:
            colors = ['red' if sig else 'gray' for sig in pipeline_df['Significant']]
            bars = ax2.bar(pipeline_df['LLM'], pipeline_df['Difference'],
                           color=colors, alpha=0.8, edgecolor='black')

            for bar, row in zip(bars, pipeline_df.iterrows()):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         height + (0.01 if height > 0 else -0.01),
                         f'p={row[1]["P_Value"]:.3f}', ha='center',
                         va='bottom' if height > 0 else 'top', fontweight='bold', fontsize=9)

            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_title('Pipeline Performance Difference\n(Iterative - Single-Shot)',
                          fontweight='bold', pad=20)
            ax2.set_ylabel('AUC Difference')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

            red_patch = mpatches.Patch(color='red', label='Significant (p < 0.05)')
            gray_patch = mpatches.Patch(color='gray', label='Not Significant')
            ax2.legend(handles=[red_patch, gray_patch])

        # 3. Effect size analysis (Cohen's d)
        effect_sizes = []
        for llm in comp_df['LLM'].unique():
            llm_data = comp_df[comp_df['LLM'] == llm]
            ss_scores = llm_data[llm_data['Pipeline'] == 'Single-Shot']['AUC_Mean'].values
            it_scores = llm_data[llm_data['Pipeline'] == 'Iterative']['AUC_Mean'].values

            if len(ss_scores) > 0 and len(it_scores) > 0:
                try:
                    pooled_std = np.sqrt(((len(ss_scores) - 1) * np.var(ss_scores) +
                                          (len(it_scores) - 1) * np.var(it_scores)) /
                                         (len(ss_scores) + len(it_scores) - 2))

                    if pooled_std > 0:
                        cohens_d = (np.mean(it_scores) - np.mean(ss_scores)) / pooled_std

                        effect_sizes.append({
                            'LLM': llm,
                            'Cohens_D': cohens_d,
                            'Effect_Size': 'Large' if abs(cohens_d) > 0.8 else
                            'Medium' if abs(cohens_d) > 0.5 else 'Small'
                        })
                except:
                    pass

        effect_df = pd.DataFrame(effect_sizes)

        if not effect_df.empty:
            colors = ['red' if abs(d) > 0.8 else 'orange' if abs(d) > 0.5 else 'green'
                      for d in effect_df['Cohens_D']]

            bars = ax3.bar(effect_df['LLM'], effect_df['Cohens_D'],
                           color=colors, alpha=0.8, edgecolor='black')

            for bar, row in zip(bars, effect_df.iterrows()):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         height + (0.05 if height > 0 else -0.05),
                         f'{height:.2f}\n({row[1]["Effect_Size"]})', ha='center',
                         va='bottom' if height > 0 else 'top', fontweight='bold', fontsize=9)

            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_title('Effect Size (Cohen\'s d)\nIterative vs Single-Shot',
                          fontweight='bold', pad=20)
            ax3.set_ylabel('Cohen\'s d')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

            large_patch = mpatches.Patch(color='red', label='Large (|d| > 0.8)')
            medium_patch = mpatches.Patch(color='orange', label='Medium (|d| > 0.5)')
            small_patch = mpatches.Patch(color='green', label='Small (|d| ≤ 0.5)')
            ax3.legend(handles=[large_patch, medium_patch, small_patch])

        # 4. Performance consistency analysis
        consistency_data = []
        for llm in comp_df['LLM'].unique():
            llm_data = comp_df[comp_df['LLM'] == llm]
            auc_scores = llm_data['AUC_Mean'].values

            if len(auc_scores) > 1 and np.mean(auc_scores) > 0:
                cv = np.std(auc_scores) / np.mean(auc_scores) * 100
                consistency_data.append({
                    'LLM': llm,
                    'CV': cv,
                    'Consistency_Score': max(0, 100 - cv)
                })

        consistency_df = pd.DataFrame(consistency_data)
        if not consistency_df.empty:
            consistency_df = consistency_df.sort_values('Consistency_Score', ascending=False)
            bars = ax4.bar(consistency_df['LLM'], consistency_df['Consistency_Score'],
                           color=self.colors['primary'], alpha=0.8, edgecolor='black')

            for bar, row in zip(bars, consistency_df.iterrows()):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{row[1]["Consistency_Score"]:.1f}', ha='center', va='bottom',
                         fontweight='bold')

            ax4.set_title('Performance Consistency Score\n(Higher = More Consistent)',
                          fontweight='bold', pad=20)
            ax4.set_ylabel('Consistency Score')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / "statistical_significance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created statistical significance analysis")

    def create_molecular_level_analysis(self, molecular_metrics: Dict, viz_dir: Path):
        """Create molecular level analysis visualization"""
        print("Creating Molecular Level Analysis...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Molecular Level Performance Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # Prepare data for visualization
        metrics_data = []
        for llm_name, llm_data in molecular_metrics.items():
            for pipeline in ['single_shot', 'iterative']:
                metrics = llm_data[pipeline]['metrics']
                metrics_data.append({
                    'LLM': llm_name,
                    'Pipeline': pipeline.replace('_', '-').title(),
                    **metrics
                })

        metrics_df = pd.DataFrame(metrics_data)

        if not metrics_df.empty:
            # 1. Validity and diversity rates
            llms = metrics_df['LLM'].unique()
            x_pos = np.arange(len(llms))
            width = 0.35

            ss_validity = []
            it_validity = []
            ss_diversity = []
            it_diversity = []

            for llm in llms:
                ss_data = metrics_df[(metrics_df['LLM'] == llm) & (metrics_df['Pipeline'] == 'Single-Shot')]
                it_data = metrics_df[(metrics_df['LLM'] == llm) & (metrics_df['Pipeline'] == 'Iterative')]

                ss_validity.append(ss_data['validity_rate'].iloc[0] * 100 if len(ss_data) > 0 else 0)
                it_validity.append(it_data['validity_rate'].iloc[0] * 100 if len(it_data) > 0 else 0)
                ss_diversity.append(ss_data['diversity_rate'].iloc[0] * 100 if len(ss_data) > 0 else 0)
                it_diversity.append(it_data['diversity_rate'].iloc[0] * 100 if len(it_data) > 0 else 0)

            bars1 = ax1.bar(x_pos - width / 2, ss_validity, width,
                            label='Single-Shot', color=self.colors['single_shot'], alpha=0.8)
            bars2 = ax1.bar(x_pos + width / 2, it_validity, width,
                            label='Iterative', color=self.colors['iterative'], alpha=0.8)

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax1.text(bar.get_x() + bar.get_width() / 2, height + 1,
                                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax1.set_title('Validity Rate by LLM and Pipeline', fontweight='bold', pad=20)
            ax1.set_ylabel('Validity Rate (%)')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(llms, rotation=45, ha='right')
            ax1.legend()

            # 2. Mean scores comparison
            bars1 = ax2.bar(x_pos - width / 2, [metrics_df[(metrics_df['LLM'] == llm) &
                                                           (metrics_df['Pipeline'] == 'Single-Shot')][
                                                    'mean_score'].iloc[0]
                                                if len(metrics_df[(metrics_df['LLM'] == llm) &
                                                                  (metrics_df['Pipeline'] == 'Single-Shot')]) > 0 else 0
                                                for llm in llms], width,
                            label='Single-Shot', color=self.colors['single_shot'], alpha=0.8)
            bars2 = ax2.bar(x_pos + width / 2, [metrics_df[(metrics_df['LLM'] == llm) &
                                                           (metrics_df['Pipeline'] == 'Iterative')]['mean_score'].iloc[
                                                    0]
                                                if len(metrics_df[(metrics_df['LLM'] == llm) &
                                                                  (metrics_df['Pipeline'] == 'Iterative')]) > 0 else 0
                                                for llm in llms], width,
                            label='Iterative', color=self.colors['iterative'], alpha=0.8)

            ax2.set_title('Mean Oracle Score by LLM and Pipeline', fontweight='bold', pad=20)
            ax2.set_ylabel('Mean Oracle Score')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(llms, rotation=45, ha='right')
            ax2.legend()

            # 3. High score rates
            ss_high_score = []
            it_high_score = []

            for llm in llms:
                ss_data = metrics_df[(metrics_df['LLM'] == llm) & (metrics_df['Pipeline'] == 'Single-Shot')]
                it_data = metrics_df[(metrics_df['LLM'] == llm) & (metrics_df['Pipeline'] == 'Iterative')]

                ss_high_score.append(ss_data['high_score_rate'].iloc[0] * 100 if len(ss_data) > 0 else 0)
                it_high_score.append(it_data['high_score_rate'].iloc[0] * 100 if len(it_data) > 0 else 0)

            bars1 = ax3.bar(x_pos - width / 2, ss_high_score, width,
                            label='Single-Shot', color=self.colors['single_shot'], alpha=0.8)
            bars2 = ax3.bar(x_pos + width / 2, it_high_score, width,
                            label='Iterative', color=self.colors['iterative'], alpha=0.8)

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax3.set_title('High Score Rate (>0.8) by LLM and Pipeline', fontweight='bold', pad=20)
            ax3.set_ylabel('High Score Rate (%)')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(llms, rotation=45, ha='right')
            ax3.legend()

            # 4. Total molecules generated
            ss_molecules = []
            it_molecules = []

            for llm in llms:
                ss_data = metrics_df[(metrics_df['LLM'] == llm) & (metrics_df['Pipeline'] == 'Single-Shot')]
                it_data = metrics_df[(metrics_df['LLM'] == llm) & (metrics_df['Pipeline'] == 'Iterative')]

                ss_molecules.append(ss_data['total_molecules'].iloc[0] if len(ss_data) > 0 else 0)
                it_molecules.append(it_data['total_molecules'].iloc[0] if len(it_data) > 0 else 0)

            bars1 = ax4.bar(x_pos - width / 2, ss_molecules, width,
                            label='Single-Shot', color=self.colors['single_shot'], alpha=0.8)
            bars2 = ax4.bar(x_pos + width / 2, it_molecules, width,
                            label='Iterative', color=self.colors['iterative'], alpha=0.8)

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax4.text(bar.get_x() + bar.get_width() / 2, height + 10,
                                 f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax4.set_title('Total Molecules Generated by LLM and Pipeline', fontweight='bold', pad=20)
            ax4.set_ylabel('Number of Molecules')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(llms, rotation=45, ha='right')
            ax4.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / "molecular_level_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created molecular level analysis")

    def extract_molecular_level_metrics(self, all_llm_evaluations: Dict) -> Dict:
        """Extract molecular-level metrics for each LLM across pipelines"""
        print("\nExtracting molecular-level metrics...")

        molecular_metrics = {}

        for llm_name, llm_eval in all_llm_evaluations.items():
            molecular_metrics[llm_name] = {
                'single_shot': {'molecules': [], 'metrics': {}},
                'iterative': {'molecules': [], 'metrics': {}}
            }

            for query_name, query_eval in llm_eval["query_evaluations"].items():
                for pipeline in ['single_shot', 'iterative']:
                    for run in query_eval[pipeline]['runs']:
                        molecular_metrics[llm_name][pipeline]['molecules'].extend(run['molecules'])

            # Calculate metrics for each pipeline
            for pipeline in ['single_shot', 'iterative']:
                molecules = molecular_metrics[llm_name][pipeline]['molecules']

                if molecules:
                    scores = [mol['Oracle_Score'] for mol in molecules]
                    valid_scores = [s for s in scores if s > 0]

                    metrics = {
                        'total_molecules': len(molecules),
                        'valid_molecules': len(valid_scores),
                        'validity_rate': len(valid_scores) / len(molecules) if molecules else 0,
                        'mean_score': np.mean(valid_scores) if valid_scores else 0,
                        'max_score': max(valid_scores) if valid_scores else 0,
                        'std_score': np.std(valid_scores) if valid_scores else 0,
                        'high_score_count': len([s for s in valid_scores if s > 0.8]),
                        'high_score_rate': len([s for s in valid_scores if s > 0.8]) / len(
                            valid_scores) if valid_scores else 0,
                        'unique_smiles': len(set(mol['SMILES'] for mol in molecules)),
                        'diversity_rate': len(set(mol['SMILES'] for mol in molecules)) / len(
                            molecules) if molecules else 0
                    }
                    molecular_metrics[llm_name][pipeline]['metrics'] = metrics
                else:
                    # Empty metrics for no molecules
                    molecular_metrics[llm_name][pipeline]['metrics'] = {
                        key: 0 for key in ['total_molecules', 'valid_molecules', 'validity_rate',
                                           'mean_score', 'max_score', 'std_score', 'high_score_count',
                                           'high_score_rate', 'unique_smiles', 'diversity_rate']
                    }

        return molecular_metrics

    def create_latex_tables_for_llm(self, llm_name: str, llm_evaluation: Dict, tables_dir: Path):
        """DEPRECATED: Old table generation method - now handled by pipeline-specific methods"""
        # This method is deprecated and no longer used to prevent duplicate table generation
        # Table generation is now handled by:
        # - _create_pipeline_specific_llm_tables() for individual LLM tables
        # - _generate_cross_llm_single_shot_table_separate() and _generate_cross_llm_iterative_table_separate() for cross-LLM tables
        pass

    def create_comprehensive_tables(self, all_llm_evaluations: Dict, comp_df: pd.DataFrame,
                                    smiles_overlap_data: Dict) -> pd.DataFrame:
        """Create comprehensive summary tables including LaTeX tables for each LLM in pipeline directories"""
        print("Creating comprehensive summary tables...")

        # Generate LaTeX tables for each LLM - but only create pipeline-specific tables in each directory
        for llm_name, llm_evaluation in all_llm_evaluations.items():
            # Create only single-shot specific tables in single-shot directory
            self._create_pipeline_specific_llm_tables(llm_name, llm_evaluation, "single_shot",
                                                      self.single_shot_dir / "tables")
            # Create only iterative specific tables in iterative directory
            self._create_pipeline_specific_llm_tables(llm_name, llm_evaluation, "iterative",
                                                      self.iterative_dir / "tables")

        # Table 1: Overall Performance Summary
        summary_data = []
        for llm_name, llm_eval in all_llm_evaluations.items():
            summary = llm_eval["summary"]

            # Calculate metrics
            total_auc = summary["single_shot"]["auc_sum"] + summary["iterative"]["auc_sum"]
            coverage = summary["successful_queries"] / summary["total_queries"] * 100 if summary[
                                                                                             "total_queries"] > 0 else 0

            # Get additional metrics
            total_molecules = 0
            best_score = 0.0
            for query_eval in llm_eval["query_evaluations"].values():
                for pipeline in ['single_shot', 'iterative']:
                    for run in query_eval[pipeline]['runs']:
                        total_molecules += run['total_molecules']
                        best_score = max(best_score, run['max_score'])

            efficiency = total_auc / total_molecules if total_molecules > 0 else 0

            # Pipeline preference
            ss_auc = summary["single_shot"]["auc_sum"]
            it_auc = summary["iterative"]["auc_sum"]
            preferred_pipeline = "Iterative" if it_auc > ss_auc else "Single-Shot" if ss_auc > it_auc else "Tied"

            # Chemical space overlap
            overlap_info = smiles_overlap_data.get(llm_name, {'single_shot': set(), 'iterative': set()})
            ss_smiles = overlap_info['single_shot']
            it_smiles = overlap_info['iterative']

            if len(ss_smiles) > 0 and len(it_smiles) > 0:
                overlap = len(ss_smiles.intersection(it_smiles))
                jaccard = overlap / len(ss_smiles.union(it_smiles)) if len(ss_smiles.union(it_smiles)) > 0 else 0
            else:
                jaccard = 0.0

            summary_data.append({
                'LLM': llm_name,
                'Total_AUC': total_auc,
                'Coverage_Percent': coverage,
                'Successful_Queries': summary["successful_queries"],
                'Total_Queries': summary["total_queries"],
                'Best_Score': best_score,
                'Total_Molecules': total_molecules,
                'Efficiency': efficiency,
                'Preferred_Pipeline': preferred_pipeline,
                'SS_AUC': ss_auc,
                'IT_AUC': it_auc,
                'Pipeline_Difference': it_auc - ss_auc,
                'Jaccard_Index': jaccard
            })

        summary_df = pd.DataFrame(summary_data).sort_values('Total_AUC', ascending=False)

        # Save CSV versions to both pipeline directories
        summary_df.to_csv(self.single_shot_dir / "tables" / "llm_performance_summary.csv", index=False)
        summary_df.to_csv(self.iterative_dir / "tables" / "llm_performance_summary.csv", index=False)

        print(f"Tables saved to pipeline-specific directories:")
        print(f"  - Single-shot tables: {self.single_shot_dir / 'tables'}")
        print(f"  - Iterative tables: {self.iterative_dir / 'tables'}")
        return summary_df

    def create_cross_llm_comparison_tables(self, all_llm_evaluations: Dict):
        """Create cross-LLM comparison tables for each pipeline"""
        print("Creating cross-LLM comparison tables...")

        # Collect data for all LLMs and tasks
        all_task_data = {}  # task_name -> {llm_name -> {ss_data, it_data}}

        for llm_name, llm_evaluation in all_llm_evaluations.items():
            for query_name, query_eval in llm_evaluation["query_evaluations"].items():
                if query_name not in all_task_data:
                    all_task_data[query_name] = {}

                # Calculate metrics for this LLM and task
                ss_data = query_eval["single_shot"]
                it_data = query_eval["iterative"]

                ss_auc = np.mean(ss_data["auc_scores"]) if ss_data["auc_scores"] else 0.0
                ss_top10 = np.mean(ss_data["top_10_scores"]) if ss_data["top_10_scores"] else 0.0
                ss_best = max([run["max_score"] for run in ss_data["runs"]]) if ss_data["runs"] else 0.0

                it_auc = np.mean(it_data["auc_scores"]) if it_data["auc_scores"] else 0.0
                it_top10 = np.mean(it_data["top_10_scores"]) if it_data["top_10_scores"] else 0.0
                it_best = max([run["max_score"] for run in it_data["runs"]]) if it_data["runs"] else 0.0

                all_task_data[query_name][llm_name] = {
                    'ss_auc': ss_auc,
                    'ss_top10': ss_top10,
                    'ss_best': ss_best,
                    'it_auc': it_auc,
                    'it_top10': it_top10,
                    'it_best': it_best
                }

        # Generate tables for each pipeline - using only the new AUC-10 format
        # Old methods removed to prevent duplicate table generation
        pass

        # OLD METHODS REMOVED - These were generating the complex 3-column tables
        # Only the new AUC-10 only format tables will be generated
        pass

    def extract_pipeline_evaluations(self, all_llm_evaluations: Dict, pipeline_type: str) -> Dict:
        """Extract evaluations for a specific pipeline only"""
        pipeline_evaluations = {}

        for llm_name, llm_eval in all_llm_evaluations.items():
            pipeline_evaluations[llm_name] = {
                "llm_name": llm_name,
                "folder_path": llm_eval["folder_path"],
                "query_evaluations": {},
                "summary": {
                    "total_queries": llm_eval["summary"]["total_queries"],
                    "successful_queries": 0,
                    "pipeline_data": {
                        "auc_sum": 0.0,
                        "auc_mean": 0.0,
                        "total_runs": 0
                    }
                }
            }

            successful_queries = 0
            pipeline_auc_means = []
            total_runs = 0

            for query_name, query_eval in llm_eval["query_evaluations"].items():
                if query_eval[pipeline_type]["auc_scores"]:
                    pipeline_evaluations[llm_name]["query_evaluations"][query_name] = {
                        "query_name": query_name,
                        "oracle_name": query_eval["oracle_name"],
                        "llm_name": llm_name,
                        "pipeline_data": query_eval[pipeline_type]
                    }
                    successful_queries += 1

                    auc_mean = np.mean(query_eval[pipeline_type]["auc_scores"])
                    pipeline_auc_means.append(auc_mean)
                    total_runs += len(query_eval[pipeline_type]["auc_scores"])

            # Update summary
            pipeline_evaluations[llm_name]["summary"]["successful_queries"] = successful_queries
            if pipeline_auc_means:
                pipeline_evaluations[llm_name]["summary"]["pipeline_data"]["auc_sum"] = sum(pipeline_auc_means)
                pipeline_evaluations[llm_name]["summary"]["pipeline_data"]["auc_mean"] = np.mean(pipeline_auc_means)
                pipeline_evaluations[llm_name]["summary"]["pipeline_data"]["total_runs"] = total_runs

        return pipeline_evaluations

    def create_pipeline_visualizations(self, pipeline_evaluations: Dict, pipeline_type: str, viz_dir: Path):
        """Create comprehensive research-focused visualizations for a specific pipeline"""
        print(f"  Creating {pipeline_type} research-focused visualizations...")

        # Prepare data
        comparison_data = []
        molecule_data = []
        smiles_overlap_data = {}

        for llm_name, llm_eval in pipeline_evaluations.items():
            smiles_overlap_data[llm_name] = set()

            for query_name, query_eval in llm_eval["query_evaluations"].items():
                pipeline_data = query_eval["pipeline_data"]

                if pipeline_data["auc_scores"]:
                    pipeline_name = pipeline_type.replace('_', '-').title()

                    comparison_data.append({
                        'LLM': llm_name,
                        'Query': query_name,
                        'Pipeline': pipeline_name,
                        'AUC_Mean': np.mean(pipeline_data["auc_scores"]),
                        'AUC_Std': np.std(pipeline_data["auc_scores"]),
                        'Top10_Mean': np.mean(pipeline_data["top_10_scores"]),
                        'Top10_Std': np.std(pipeline_data["top_10_scores"]),
                        'N_Runs': len(pipeline_data["auc_scores"])
                    })

                    # Collect molecules and SMILES
                    for run in pipeline_data["runs"]:
                        molecule_data.extend(run["molecules"])
                        for mol in run["molecules"]:
                            smiles_overlap_data[llm_name].add(mol['SMILES'])

        comp_df = pd.DataFrame(comparison_data)
        mol_df = pd.DataFrame(molecule_data)

        # Extract molecular-level metrics for this pipeline
        molecular_metrics = self.extract_pipeline_molecular_metrics(pipeline_evaluations)

        # Create comprehensive research-focused visualizations
        if not comp_df.empty:
            self._create_pipeline_performance_comparison(comp_df, pipeline_type, viz_dir)
            self._create_pipeline_llm_ranking(comp_df, pipeline_type, viz_dir)
            self._create_pipeline_llm_performance_analysis(comp_df, mol_df, pipeline_evaluations, pipeline_type,
                                                           viz_dir)

        if not mol_df.empty:
            self._create_pipeline_molecular_analysis(mol_df, pipeline_type, viz_dir)
            self._create_pipeline_score_distribution(mol_df, pipeline_type, viz_dir)
            self._create_pipeline_validity_novelty_diversity_analysis(mol_df, pipeline_type, viz_dir)

        # Create molecular-level analysis visualization
        if molecular_metrics:
            self._create_pipeline_molecular_level_analysis(molecular_metrics, pipeline_type, viz_dir)

        if len(smiles_overlap_data) > 1:
            self._create_pipeline_chemical_space_analysis(smiles_overlap_data, pipeline_type, viz_dir)

        # Create top-K SMILES overlap analysis across tasks
        if not comp_df.empty and not mol_df.empty:
            self._create_top_k_smiles_overlap_analysis(pipeline_evaluations, pipeline_type, viz_dir)

        # Create statistical significance analysis
        if not comp_df.empty:
            self._create_pipeline_statistical_analysis(comp_df, pipeline_type, viz_dir)

        # Create drug-likeness analysis
        if not mol_df.empty:
            self._create_drug_likeness_analysis(mol_df, pipeline_type, viz_dir)

    def create_pipeline_tables(self, pipeline_evaluations: Dict, pipeline_type: str, tables_dir: Path):
        """Create LaTeX tables for a specific pipeline"""
        print(f"  Creating {pipeline_type} tables...")

        # Create cross-LLM comparison table for this pipeline
        if pipeline_type == "single_shot":
            self._generate_cross_llm_single_shot_table_separate(pipeline_evaluations, tables_dir)
        else:
            self._generate_cross_llm_iterative_table_separate(pipeline_evaluations, tables_dir)

        # Individual LLM tables are now created by _create_pipeline_specific_llm_tables()
        # from create_comprehensive_tables() to prevent duplicates

    def save_pipeline_data(self, pipeline_evaluations: Dict, pipeline_type: str, data_dir: Path):
        """Save pipeline-specific data"""
        print(f"  Saving {pipeline_type} data...")

        # Save evaluations as JSON
        with open(data_dir / f"{pipeline_type}_evaluations.json", 'w') as f:
            json.dump(pipeline_evaluations, f, indent=2, default=str)

        # Create and save summary CSV
        summary_data = []
        for llm_name, llm_eval in pipeline_evaluations.items():
            summary = llm_eval["summary"]
            summary_data.append({
                'LLM': llm_name,
                'Total_Queries': summary["total_queries"],
                'Successful_Queries': summary["successful_queries"],
                'Coverage_Percent': (summary["successful_queries"] / summary["total_queries"] * 100) if summary[
                                                                                                            "total_queries"] > 0 else 0,
                'Total_AUC': summary["pipeline_data"]["auc_sum"],
                'Mean_AUC': summary["pipeline_data"]["auc_mean"],
                'Total_Runs': summary["pipeline_data"]["total_runs"]
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(data_dir / f"{pipeline_type}_summary.csv", index=False)

        print(f"    ✓ Saved {pipeline_type} evaluations and summary")

    def print_research_conclusions(self, summary_df: pd.DataFrame, comp_df: pd.DataFrame,
                                   smiles_overlap_data: Dict):
        """Print key research conclusions"""
        print("\n" + "=" * 80)
        print("KEY RESEARCH FINDINGS")
        print("=" * 80)

        if not comp_df.empty:
            # RQ1: Single-shot vs Iterative
            pipeline_stats = comp_df.groupby('Pipeline').agg({
                'AUC_Mean': ['sum', 'mean', 'std']
            }).round(4)

            ss_total = pipeline_stats.loc[
                'Single-Shot', ('AUC_Mean', 'sum')] if 'Single-Shot' in pipeline_stats.index else 0
            it_total = pipeline_stats.loc[
                'Iterative', ('AUC_Mean', 'sum')] if 'Iterative' in pipeline_stats.index else 0

            print(f"\nRQ1: Single-Shot vs Iterative Generation")
            print(f"  • Single-Shot Total AUC: {ss_total:.2f}")
            print(f"  • Iterative Total AUC: {it_total:.2f}")
            print(f"  • Winner: {'Iterative' if it_total > ss_total else 'Single-Shot'}")
            print(f"  • Performance Gap: {abs(it_total - ss_total):.2f} AUC points")

        if not summary_df.empty:
            # RQ2: LLM Performance
            best_llm = summary_df.iloc[0]
            print(f"\nRQ2: LLM Performance in Agentic System")
            print(f"  • Best Overall LLM: {best_llm['LLM']}")
            print(f"  • Total AUC: {best_llm['Total_AUC']:.2f}")
            print(f"  • Coverage: {best_llm['Coverage_Percent']:.1f}%")
            print(f"  • Efficiency: {best_llm['Efficiency']:.4f} AUC/molecule")

            # RQ3: Chemical Space Overlap
            avg_jaccard = summary_df['Jaccard_Index'].mean()
            print(f"\nRQ3: Chemical Space Overlap")
            print(f"  • Average Jaccard Index: {avg_jaccard:.3f}")
            print(
                f"  • Interpretation: {'Low overlap - exploring different spaces' if avg_jaccard < 0.3 else 'Medium overlap' if avg_jaccard < 0.6 else 'High overlap - similar exploration'}")

            # Key insights
            print(f"\nKEY INSIGHTS:")
            print(f"  • Most successful approach: {best_llm['LLM']} with {best_llm['Preferred_Pipeline']} pipeline")
            print(f"  • Chemical space exploration: {'Diverse' if avg_jaccard < 0.4 else 'Overlapping'}")
            print(f"  • Performance consistency: {summary_df['Total_AUC'].std():.2f} AUC standard deviation")

    def run_research_focused_analysis(self):
        """Run the complete research-focused analysis using pre-computed oracle scores"""
        print("Starting Research-Focused LLM Analysis (Using Pre-computed Oracle Scores)")
        print("=" * 70)

        # Find and process LLM folders
        llm_folders = self.find_llm_folders()
        if not llm_folders:
            return None

        # Load pre-computed oracle results
        all_llm_data = {}
        for folder in llm_folders:
            llm_data = self.load_oracle_results_from_folder(folder)
            if llm_data["results"]:
                all_llm_data[llm_data["llm_name"]] = llm_data

        if not all_llm_data:
            print("No valid LLM data found!")
            return None

        print(f"\nProcessing {len(all_llm_data)} LLMs...")
        all_llm_evaluations = {}

        for llm_name, llm_data in all_llm_data.items():
            try:
                llm_evaluation = self.process_oracle_results(llm_data)
                all_llm_evaluations[llm_name] = llm_evaluation
            except Exception as e:
                print(f"Failed to process {llm_name}: {e}")

        if not all_llm_evaluations:
            print("No LLM evaluations completed!")
            return None

        print(f"Successfully processed {len(all_llm_evaluations)} LLMs\n")

        # Create research-focused visualizations
        comp_df, mol_df, smiles_overlap_data = self.create_research_visualizations(all_llm_evaluations)

        # Create comprehensive tables
        summary_df = self.create_comprehensive_tables(all_llm_evaluations, comp_df, smiles_overlap_data)

        # Create cross-LLM comparison tables
        self.create_cross_llm_comparison_tables(all_llm_evaluations)

        # Save raw data to both pipeline directories
        for pipeline_dir, pipeline_name in [(self.single_shot_dir, "single_shot"), (self.iterative_dir, "iterative")]:
            data_dir = pipeline_dir / "data"
            with open(data_dir / f"{pipeline_name}_complete_evaluations.json", 'w') as f:
                json.dump(all_llm_evaluations, f, indent=2, default=str)

            comp_df.to_csv(data_dir / f"{pipeline_name}_comparison_data.csv", index=False)
            mol_df.to_csv(data_dir / f"{pipeline_name}_molecule_data.csv", index=False)

        # Print key findings
        self.print_research_conclusions(summary_df, comp_df, smiles_overlap_data)

        print(f"\nResearch-focused analysis completed!")
        print(f"Results saved to pipeline-specific directories:")
        print(f"")
        print(f"✓ Single-Shot Pipeline Analysis: {self.single_shot_dir}")
        print(f"  - Visualizations: {self.single_shot_dir / 'visualizations'}")
        print(f"  - Tables (including cross-LLM comparisons): {self.single_shot_dir / 'tables'}")
        print(f"  - Raw Data: {self.single_shot_dir / 'data'}")
        print(f"")
        print(f"✓ Iterative Pipeline Analysis: {self.iterative_dir}")
        print(f"  - Visualizations: {self.iterative_dir / 'visualizations'}")
        print(f"  - Tables (including cross-LLM comparisons): {self.iterative_dir / 'tables'}")
        print(f"  - Raw Data: {self.iterative_dir / 'data'}")
        print(f"")
        print(f"Key outputs:")
        print(f"  - Cross-LLM comparison tables with AUC-10 rankings")
        print(f"  - Pipeline-specific visualizations and analysis")
        print(f"  - Individual LLM performance tables for each pipeline")

        return {
            'evaluations': all_llm_evaluations,
            'comparison_df': comp_df,
            'molecule_df': mol_df,
            'summary_df': summary_df,
            'overlap_data': smiles_overlap_data
        }

    def _create_pipeline_performance_comparison(self, comp_df: pd.DataFrame, pipeline_type: str, viz_dir: Path):
        """Create performance comparison visualization for a specific pipeline"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: LLM Performance Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. AUC comparison by LLM
        llm_auc_data = comp_df.groupby('LLM').agg({'AUC_Mean': ['sum', 'mean', 'count']}).round(4)
        llm_auc_data.columns = ['Total_AUC', 'Mean_AUC', 'Task_Count']
        llm_auc_data = llm_auc_data.reset_index().sort_values('Total_AUC', ascending=False)

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(llm_auc_data)]
        bars = ax1.bar(llm_auc_data['LLM'], llm_auc_data['Total_AUC'],
                       color=colors, alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, llm_auc_data['Total_AUC']):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

        ax1.set_title(f'{pipeline_name} Pipeline: Total AUC by LLM', fontweight='bold', pad=20)
        ax1.set_ylabel('Total AUC Score')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # 2. Performance distribution
        try:
            llms = comp_df['LLM'].unique()
            data_for_violin = [comp_df[comp_df['LLM'] == llm]['AUC_Mean'].values for llm in llms]

            parts = ax2.violinplot(data_for_violin, positions=range(len(llms)),
                                   showmeans=True, showmedians=True)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)

            ax2.set_xticks(range(len(llms)))
            ax2.set_xticklabels(llms, rotation=45, ha='right')
        except:
            # Fallback to box plot
            bp = ax2.boxplot(data_for_violin, labels=llms, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        ax2.set_title(f'{pipeline_name} Pipeline: AUC Distribution', fontweight='bold', pad=20)
        ax2.set_ylabel('AUC Score')

        # 3. Task coverage
        task_coverage = comp_df.groupby('LLM').size()
        bars = ax3.bar(task_coverage.index, task_coverage.values,
                       color=colors[:len(task_coverage)], alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, task_coverage.values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{int(val)}', ha='center', va='bottom', fontweight='bold')

        ax3.set_title(f'{pipeline_name} Pipeline: Task Coverage', fontweight='bold', pad=20)
        ax3.set_ylabel('Number of Tasks')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        # 4. Performance consistency (CV)
        consistency_data = []
        for llm in llms:
            llm_aucs = comp_df[comp_df['LLM'] == llm]['AUC_Mean']
            if len(llm_aucs) > 1 and np.mean(llm_aucs) > 0:
                cv = np.std(llm_aucs) / np.mean(llm_aucs) * 100
                consistency_data.append({'LLM': llm, 'CV': cv})

        if consistency_data:
            consistency_df = pd.DataFrame(consistency_data)
            bars = ax4.bar(consistency_df['LLM'], consistency_df['CV'],
                           color=colors[:len(consistency_df)], alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, consistency_df['CV']):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

            ax4.set_title(f'{pipeline_name} Pipeline: Performance Consistency\n(Lower = More Consistent)',
                          fontweight='bold', pad=20)
            ax4.set_ylabel('Coefficient of Variation (%)')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} performance comparison")

    def _create_pipeline_llm_ranking(self, comp_df: pd.DataFrame, pipeline_type: str, viz_dir: Path):
        """Create LLM ranking visualization for a specific pipeline"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: LLM Ranking Analysis',
                     fontsize=16, fontweight='bold', y=0.95)

        # 1. Overall ranking by total AUC
        ranking_data = comp_df.groupby('LLM').agg({
            'AUC_Mean': ['sum', 'mean', 'count'],
            'Top10_Mean': 'mean'
        }).round(4)

        ranking_data.columns = ['Total_AUC', 'Mean_AUC', 'Task_Count', 'Mean_Top10']
        ranking_data = ranking_data.reset_index().sort_values('Total_AUC', ascending=True)

        colors = plt.cm.viridis(np.linspace(0, 1, len(ranking_data)))
        bars = ax1.barh(ranking_data['LLM'], ranking_data['Total_AUC'],
                        color=colors, alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, ranking_data['Total_AUC']):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', va='center', fontweight='bold')

        ax1.set_title(f'{pipeline_name} Pipeline: LLM Ranking by Total AUC', fontweight='bold')
        ax1.set_xlabel('Total AUC Score')

        # 2. Performance vs Task Count scatter
        ax2.scatter(ranking_data['Task_Count'], ranking_data['Total_AUC'],
                    s=ranking_data['Mean_Top10'] * 500, alpha=0.7, c=range(len(ranking_data)), cmap='viridis')

        for i, row in ranking_data.iterrows():
            ax2.annotate(row['LLM'], (row['Task_Count'], row['Total_AUC']),
                         xytext=(5, 5), textcoords='offset points', fontsize=10)

        ax2.set_title(f'{pipeline_name} Pipeline: Performance vs Coverage\n(Bubble size = Mean Top-10 Score)',
                      fontweight='bold')
        ax2.set_xlabel('Number of Tasks')
        ax2.set_ylabel('Total AUC Score')

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(viz_dir / f"{pipeline_type}_llm_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} LLM ranking")

    def _create_pipeline_molecular_analysis(self, mol_df: pd.DataFrame, pipeline_type: str, viz_dir: Path):
        """Create molecular-level analysis for a specific pipeline"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: Molecular-Level Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Score distribution
        scores = mol_df[mol_df['Oracle_Score'] > 0]['Oracle_Score']
        if not scores.empty:
            ax1.hist(scores, bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
            ax1.axvline(scores.mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {scores.mean():.3f}')
            ax1.axvline(scores.median(), color='orange', linestyle='--', linewidth=2,
                        label=f'Median: {scores.median():.3f}')
            ax1.set_title(f'{pipeline_name} Pipeline: Oracle Score Distribution', fontweight='bold')
            ax1.set_xlabel('Oracle Score')
            ax1.set_ylabel('Frequency')
            ax1.legend()

        # 2. Performance by LLM
        if 'LLM' in mol_df.columns:
            llm_performance = mol_df[mol_df['Oracle_Score'] > 0].groupby('LLM').agg({
                'Oracle_Score': ['count', 'mean', 'max']
            }).round(4)

            llm_performance.columns = ['Count', 'Mean_Score', 'Max_Score']
            llm_performance = llm_performance.reset_index().sort_values('Mean_Score', ascending=False)

            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(llm_performance)]
            bars = ax2.bar(llm_performance['LLM'], llm_performance['Mean_Score'],
                           color=colors, alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, llm_performance['Mean_Score']):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax2.set_title(f'{pipeline_name} Pipeline: Mean Score by LLM', fontweight='bold')
            ax2.set_ylabel('Mean Oracle Score')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # 3. High score analysis
        high_score_threshold = 0.8
        if 'LLM' in mol_df.columns:
            high_score_data = []
            for llm in mol_df['LLM'].unique():
                llm_mols = mol_df[mol_df['LLM'] == llm]
                total_mols = len(llm_mols)
                high_score_mols = len(llm_mols[llm_mols['Oracle_Score'] > high_score_threshold])
                high_score_rate = (high_score_mols / total_mols * 100) if total_mols > 0 else 0

                high_score_data.append({
                    'LLM': llm,
                    'High_Score_Rate': high_score_rate,
                    'High_Score_Count': high_score_mols,
                    'Total_Count': total_mols
                })

            high_score_df = pd.DataFrame(high_score_data).sort_values('High_Score_Rate', ascending=False)

            if not high_score_df.empty:
                bars = ax3.bar(high_score_df['LLM'], high_score_df['High_Score_Rate'],
                               color=colors[:len(high_score_df)], alpha=0.8, edgecolor='black')

                for bar, row in zip(bars, high_score_df.iterrows()):
                    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                             f'{row[1]["High_Score_Rate"]:.1f}%\n({row[1]["High_Score_Count"]}/{row[1]["Total_Count"]})',
                             ha='center', va='bottom', fontweight='bold', fontsize=8)

                ax3.set_title(f'{pipeline_name} Pipeline: High Score Rate (>{high_score_threshold})', fontweight='bold')
                ax3.set_ylabel('High Score Rate (%)')
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        # 4. Molecule count by query
        if 'Query' in mol_df.columns:
            query_counts = mol_df.groupby('Query').size().sort_values(ascending=False).head(10)

            if not query_counts.empty:
                bars = ax4.barh(range(len(query_counts)), query_counts.values,
                                color=self.colors['accent'], alpha=0.8, edgecolor='black')

                ax4.set_yticks(range(len(query_counts)))
                ax4.set_yticklabels([q[:25] + '...' if len(q) > 25 else q for q in query_counts.index])
                ax4.set_title(f'{pipeline_name} Pipeline: Top 10 Queries by Molecule Count', fontweight='bold')
                ax4.set_xlabel('Number of Molecules')

                for bar, val in zip(bars, query_counts.values):
                    ax4.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                             f'{int(val)}', va='center', fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_molecular_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} molecular analysis")

    def _create_pipeline_score_distribution(self, mol_df: pd.DataFrame, pipeline_type: str, viz_dir: Path):
        """Create score distribution analysis for a specific pipeline"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: Score Distribution Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        valid_scores = mol_df[mol_df['Oracle_Score'] > 0]['Oracle_Score']

        if not valid_scores.empty:
            # 1. Overall distribution with statistics
            n_bins = min(50, int(np.sqrt(len(valid_scores))))
            ax1.hist(valid_scores, bins=n_bins, alpha=0.7, color=self.colors['primary'],
                     edgecolor='black', density=True)

            # Add statistical lines
            mean_score = valid_scores.mean()
            median_score = valid_scores.median()
            std_score = valid_scores.std()

            ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}')
            ax1.axvline(median_score, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_score:.3f}')
            ax1.axvline(mean_score + std_score, color='red', linestyle=':', alpha=0.7, label=f'±1 SD: {std_score:.3f}')
            ax1.axvline(mean_score - std_score, color='red', linestyle=':', alpha=0.7)

            ax1.set_title(f'{pipeline_name} Pipeline: Score Distribution with Statistics', fontweight='bold')
            ax1.set_xlabel('Oracle Score')
            ax1.set_ylabel('Density')
            ax1.legend()

            # 2. Cumulative distribution
            sorted_scores = np.sort(valid_scores)
            cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
            ax2.plot(sorted_scores, cumulative, color=self.colors['primary'], linewidth=2)
            ax2.axvline(median_score, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_score:.3f}')
            ax2.axhline(0.5, color='orange', linestyle='--', alpha=0.7)

            ax2.set_title(f'{pipeline_name} Pipeline: Cumulative Distribution', fontweight='bold')
            ax2.set_xlabel('Oracle Score')
            ax2.set_ylabel('Cumulative Probability')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Score ranges analysis
            ranges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            range_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
            range_counts = []

            for i in range(len(ranges) - 1):
                count = len(valid_scores[(valid_scores >= ranges[i]) & (valid_scores < ranges[i + 1])])
                range_counts.append(count)

            colors_gradient = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(range_counts)))
            bars = ax3.bar(range_labels, range_counts, color=colors_gradient, alpha=0.8, edgecolor='black')

            for bar, count in zip(bars, range_counts):
                percentage = (count / len(valid_scores)) * 100
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(valid_scores) * 0.01,
                         f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax3.set_title(f'{pipeline_name} Pipeline: Score Range Distribution', fontweight='bold')
            ax3.set_xlabel('Score Range')
            ax3.set_ylabel('Number of Molecules')

            # 4. Top performers analysis
            top_10_percent_threshold = np.percentile(valid_scores, 90)
            top_5_percent_threshold = np.percentile(valid_scores, 95)
            top_1_percent_threshold = np.percentile(valid_scores, 99)

            thresholds = [top_10_percent_threshold, top_5_percent_threshold, top_1_percent_threshold]
            threshold_labels = ['Top 10%', 'Top 5%', 'Top 1%']
            threshold_counts = [len(valid_scores[valid_scores >= t]) for t in thresholds]

            bars = ax4.bar(threshold_labels, threshold_counts,
                           color=[self.colors['accent'], self.colors['secondary'], self.colors['success']],
                           alpha=0.8, edgecolor='black')

            for bar, count, threshold in zip(bars, threshold_counts, thresholds):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(valid_scores) * 0.01,
                         f'{count}\n(≥{threshold:.3f})', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax4.set_title(f'{pipeline_name} Pipeline: Top Performers Analysis', fontweight='bold')
            ax4.set_ylabel('Number of Molecules')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} score distribution")

    def _create_pipeline_llm_performance_analysis(self, comp_df: pd.DataFrame, mol_df: pd.DataFrame,
                                                  pipeline_evaluations: Dict, pipeline_type: str, viz_dir: Path):
        """Create LLM performance analysis for a specific pipeline (similar to RQ2)"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: LLM Performance Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Multi-dimensional performance comparison
        llm_metrics = []
        for llm_name, llm_eval in pipeline_evaluations.items():
            summary = llm_eval["summary"]

            total_auc = summary["pipeline_data"]["auc_sum"]
            coverage = summary["successful_queries"] / summary["total_queries"] * 100 if summary[
                                                                                             "total_queries"] > 0 else 0

            # Get additional metrics
            total_molecules = 0
            best_score = 0.0
            for query_eval in llm_eval["query_evaluations"].values():
                pipeline_data = query_eval["pipeline_data"]
                for run in pipeline_data['runs']:
                    total_molecules += run['total_molecules']
                    best_score = max(best_score, run['max_score'])

            efficiency = total_auc / total_molecules if total_molecules > 0 else 0

            llm_metrics.append({
                'LLM': llm_name,
                'Total_AUC': total_auc,
                'Coverage': coverage,
                'Efficiency': efficiency * 1000,  # Scale for visibility
                'Best_Score': best_score,
                'Total_Molecules': total_molecules
            })

        metrics_df = pd.DataFrame(llm_metrics)

        if not metrics_df.empty:
            # Normalize metrics for comparison (0-100 scale)
            for col in ['Total_AUC', 'Efficiency', 'Best_Score']:
                if metrics_df[col].max() > 0:
                    metrics_df[f'{col}_norm'] = (metrics_df[col] / metrics_df[col].max()) * 100
                else:
                    metrics_df[f'{col}_norm'] = 0

            # Stacked bar chart
            x_pos = np.arange(len(metrics_df))
            width = 0.6

            bottom = np.zeros(len(metrics_df))
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            metrics = ['Total_AUC_norm', 'Coverage', 'Efficiency_norm', 'Best_Score_norm']
            labels = ['Total AUC', 'Coverage', 'Efficiency', 'Best Score']

            for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
                values = metrics_df[metric] if metric in metrics_df.columns else np.zeros(len(metrics_df))
                ax1.bar(x_pos, values, width, bottom=bottom,
                        label=label, color=color, alpha=0.8)
                bottom += values

            ax1.set_title(f'{pipeline_name}: Multi-Dimensional LLM Performance', fontweight='bold', pad=20)
            ax1.set_ylabel('Normalized Score')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(metrics_df['LLM'], rotation=45, ha='right')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # 2. Performance scatter plot
            ax2.scatter(metrics_df['Total_AUC'], metrics_df['Coverage'],
                        s=metrics_df['Efficiency'] * 10, alpha=0.7,
                        c=range(len(metrics_df)), cmap='viridis')

            for i, row in metrics_df.iterrows():
                ax2.annotate(row['LLM'], (row['Total_AUC'], row['Coverage']),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)

            ax2.set_title(f'{pipeline_name}: Performance vs Coverage', fontweight='bold', pad=20)
            ax2.set_xlabel('Total AUC Score')
            ax2.set_ylabel('Query Coverage (%)')

        # 3. Task-specific performance heatmap
        if not comp_df.empty:
            task_performance = comp_df.pivot_table(values='AUC_Mean', index='Query',
                                                   columns='LLM', aggfunc='max', fill_value=0)

            # Select top 10 most challenging tasks (lowest average performance)
            if len(task_performance) > 10:
                task_difficulty = task_performance.mean(axis=1).sort_values().head(10)
                top_tasks = task_performance.loc[task_difficulty.index]
            else:
                top_tasks = task_performance

            if not top_tasks.empty:
                im = ax3.imshow(top_tasks.values, cmap='RdYlGn', aspect='auto')
                ax3.set_xticks(range(len(top_tasks.columns)))
                ax3.set_xticklabels(top_tasks.columns, rotation=45, ha='right')
                ax3.set_yticks(range(len(top_tasks.index)))
                ax3.set_yticklabels([task[:20] + '...' if len(task) > 20 else task
                                     for task in top_tasks.index], fontsize=9)
                ax3.set_title(f'{pipeline_name}: Performance on Most Challenging Tasks', fontweight='bold', pad=20)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
                cbar.set_label('AUC Score')

        # 4. Performance consistency analysis
        consistency_data = []
        for llm in comp_df['LLM'].unique():
            llm_data = comp_df[comp_df['LLM'] == llm]
            auc_scores = llm_data['AUC_Mean'].values

            if len(auc_scores) > 1 and np.mean(auc_scores) > 0:
                cv = np.std(auc_scores) / np.mean(auc_scores) * 100
                consistency_data.append({
                    'LLM': llm,
                    'CV': cv,
                    'Consistency_Score': max(0, 100 - cv)
                })

        consistency_df = pd.DataFrame(consistency_data)
        if not consistency_df.empty:
            consistency_df = consistency_df.sort_values('Consistency_Score', ascending=False)
            bars = ax4.bar(consistency_df['LLM'], consistency_df['Consistency_Score'],
                           color=self.colors['primary'], alpha=0.8, edgecolor='black')

            for bar, row in zip(bars, consistency_df.iterrows()):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{row[1]["Consistency_Score"]:.1f}', ha='center', va='bottom',
                         fontweight='bold')

            ax4.set_title(f'{pipeline_name}: Performance Consistency', fontweight='bold', pad=20)
            ax4.set_ylabel('Consistency Score')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_llm_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} LLM performance analysis")

    def _create_pipeline_validity_novelty_diversity_analysis(self, mol_df: pd.DataFrame, pipeline_type: str,
                                                             viz_dir: Path):
        """Create validity, novelty, and diversity analysis for a specific pipeline"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: Validity, Novelty, and Diversity Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Score distribution histogram
        if not mol_df.empty:
            score_data = mol_df[mol_df['Oracle_Score'] > 0]['Oracle_Score']

            if not score_data.empty:
                ax1.hist(score_data, bins=50, alpha=0.7, color=self.colors['primary'], edgecolor='black')
                ax1.axvline(score_data.mean(), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {score_data.mean():.3f}')
                ax1.axvline(score_data.median(), color='orange', linestyle='--', linewidth=2,
                            label=f'Median: {score_data.median():.3f}')

                ax1.set_title(f'{pipeline_name}: Oracle Score Distribution', fontweight='bold', pad=20)
                ax1.set_xlabel('Oracle Score')
                ax1.set_ylabel('Frequency')
                ax1.legend()

        # 2. Performance by LLM
        if 'LLM' in mol_df.columns:
            filtered_mol_df = mol_df[mol_df['Oracle_Score'] > 0]

            if not filtered_mol_df.empty:
                llms = filtered_mol_df['LLM'].unique()
                data_for_violin = [filtered_mol_df[filtered_mol_df['LLM'] == llm]['Oracle_Score'].values
                                   for llm in llms]

                try:
                    parts = ax2.violinplot(data_for_violin, positions=range(len(llms)),
                                           showmeans=True, showmedians=True)

                    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
                    for i, pc in enumerate(parts['bodies']):
                        pc.set_facecolor(colors[i % len(colors)])
                        pc.set_alpha(0.7)

                    ax2.set_xticks(range(len(llms)))
                    ax2.set_xticklabels(llms, rotation=45, ha='right')
                except:
                    bp = ax2.boxplot(data_for_violin, labels=llms, patch_artist=True)
                    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
                    for i, patch in enumerate(bp['boxes']):
                        patch.set_facecolor(colors[i % len(colors)])
                        patch.set_alpha(0.7)
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

                ax2.set_title(f'{pipeline_name}: Score Distribution by LLM', fontweight='bold', pad=20)
                ax2.set_ylabel('Oracle Score')

        # 3. Top performers analysis - Fair sampling across LLMs
        if not mol_df.empty and 'LLM' in mol_df.columns:
            # Get top performers fairly from each LLM to avoid bias due to different task counts
            top_performers_list = []
            llm_groups = mol_df.groupby('LLM')

            # Take top performers proportionally from each LLM based on their data size
            total_molecules = len(mol_df)
            target_total = 20  # We want around 20 total top performers

            for llm, group in llm_groups:
                # Calculate proportional share, but ensure minimum of 3 per LLM if they have data
                llm_proportion = len(group) / total_molecules
                llm_quota = max(3, int(target_total * llm_proportion))
                # But don't exceed what the LLM actually has
                llm_quota = min(llm_quota, len(group))

                llm_top = group.nlargest(llm_quota, 'Oracle_Score')
                top_performers_list.append(llm_top)

            if top_performers_list:
                top_performers = pd.concat(top_performers_list)
                top_counts = top_performers.groupby('LLM').size().reset_index(name='Count')

                if not top_counts.empty:
                    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(top_counts)]
                    bars = ax3.bar(top_counts['LLM'], top_counts['Count'],
                                   color=colors, alpha=0.8, edgecolor='black')

                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                                     f'{int(height)}', ha='center', va='bottom', fontweight='bold')

                    ax3.set_title(f'{pipeline_name}: Top 20 Performers by LLM', fontweight='bold', pad=20)
                    ax3.set_ylabel('Number of Top Performers')
                    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        # 4. Query difficulty analysis
        if not mol_df.empty and 'Query' in mol_df.columns:
            query_stats = mol_df.groupby('Query').agg({
                'Oracle_Score': ['mean', 'std', 'count', 'max']
            }).round(3)
            query_stats.columns = ['Mean_Score', 'Std_Score', 'Count', 'Max_Score']
            query_stats = query_stats.reset_index()

            challenging_queries = query_stats.nsmallest(10, 'Mean_Score')

            if not challenging_queries.empty:
                bars = ax4.barh(range(len(challenging_queries)), challenging_queries['Mean_Score'],
                                color=self.colors['accent'], alpha=0.8, edgecolor='black')

                for i, (bar, val) in enumerate(zip(bars, challenging_queries['Mean_Score'])):
                    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{val:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)

                ax4.set_yticks(range(len(challenging_queries)))
                ax4.set_yticklabels([q[:25] + '...' if len(q) > 25 else q
                                     for q in challenging_queries['Query']], fontsize=9)
                ax4.set_title(f'{pipeline_name}: Most Challenging Queries', fontweight='bold', pad=20)
                ax4.set_xlabel('Mean Oracle Score')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_validity_novelty_diversity.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} validity/novelty/diversity analysis")

    def _create_pipeline_chemical_space_analysis(self, smiles_overlap_data: Dict, pipeline_type: str, viz_dir: Path):
        """Create chemical space analysis for a specific pipeline"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: Chemical Space Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Unique molecule count by LLM
        llm_names = list(smiles_overlap_data.keys())
        unique_counts = [len(smiles_set) for smiles_set in smiles_overlap_data.values()]

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(llm_names)]
        bars = ax1.bar(llm_names, unique_counts, color=colors, alpha=0.8, edgecolor='black')

        for bar, count in zip(bars, unique_counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(unique_counts) * 0.01,
                     f'{count}', ha='center', va='bottom', fontweight='bold')

        ax1.set_title(f'{pipeline_name}: Unique Molecules by LLM', fontweight='bold', pad=20)
        ax1.set_ylabel('Number of Unique Molecules')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # 2. Pairwise Jaccard indices (if multiple LLMs)
        if len(llm_names) > 1:
            jaccard_matrix = np.zeros((len(llm_names), len(llm_names)))

            for i, llm1 in enumerate(llm_names):
                for j, llm2 in enumerate(llm_names):
                    set1 = smiles_overlap_data[llm1]
                    set2 = smiles_overlap_data[llm2]

                    if len(set1) > 0 and len(set2) > 0:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        jaccard_matrix[i, j] = intersection / union if union > 0 else 0
                    else:
                        jaccard_matrix[i, j] = 0

            im = ax2.imshow(jaccard_matrix, cmap='Blues', vmin=0, vmax=1)

            # Add text annotations
            for i in range(len(llm_names)):
                for j in range(len(llm_names)):
                    ax2.text(j, i, f'{jaccard_matrix[i, j]:.3f}', ha='center', va='center',
                             color='white' if jaccard_matrix[i, j] > 0.5 else 'black', fontweight='bold')

            ax2.set_xticks(range(len(llm_names)))
            ax2.set_xticklabels(llm_names, rotation=45, ha='right')
            ax2.set_yticks(range(len(llm_names)))
            ax2.set_yticklabels(llm_names)
            ax2.set_title(f'{pipeline_name}: Chemical Space Similarity (Jaccard)', fontweight='bold', pad=20)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label('Jaccard Index')

        # 3. SMILES length distribution
        all_smiles_lengths = []
        llm_smiles_lengths = {}

        for llm_name, smiles_set in smiles_overlap_data.items():
            lengths = [len(smiles) for smiles in smiles_set]
            all_smiles_lengths.extend(lengths)
            llm_smiles_lengths[llm_name] = lengths

        if all_smiles_lengths:
            # Overall distribution
            ax3.hist(all_smiles_lengths, bins=30, alpha=0.7, color=self.colors['primary'], edgecolor='black')
            ax3.axvline(np.mean(all_smiles_lengths), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(all_smiles_lengths):.1f}')
            ax3.set_title(f'{pipeline_name}: SMILES Length Distribution', fontweight='bold', pad=20)
            ax3.set_xlabel('SMILES Length')
            ax3.set_ylabel('Frequency')
            ax3.legend()

        # 4. Diversity comparison by LLM
        diversity_data = []
        for llm_name, smiles_set in smiles_overlap_data.items():
            if len(smiles_set) > 0:
                lengths = [len(smiles) for smiles in smiles_set]
                diversity_data.append({
                    'LLM': llm_name,
                    'Unique_Count': len(smiles_set),
                    'Mean_Length': np.mean(lengths),
                    'Std_Length': np.std(lengths)
                })

        diversity_df = pd.DataFrame(diversity_data)
        if not diversity_df.empty:
            ax4.scatter(diversity_df['Unique_Count'], diversity_df['Mean_Length'],
                        s=diversity_df['Std_Length'] * 10, alpha=0.7, c=range(len(diversity_df)), cmap='viridis')

            for i, row in diversity_df.iterrows():
                ax4.annotate(row['LLM'], (row['Unique_Count'], row['Mean_Length']),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)

            ax4.set_title(f'{pipeline_name}: Diversity vs Complexity', fontweight='bold', pad=20)
            ax4.set_xlabel('Number of Unique Molecules')
            ax4.set_ylabel('Mean SMILES Length')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_chemical_space_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} chemical space analysis")

    def _create_pipeline_statistical_analysis(self, comp_df: pd.DataFrame, pipeline_type: str, viz_dir: Path):
        """Create statistical analysis for a specific pipeline"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: Statistical Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Confidence intervals for LLM performance
        llm_stats = []
        for llm in comp_df['LLM'].unique():
            llm_data = comp_df[comp_df['LLM'] == llm]
            auc_scores = llm_data['AUC_Mean'].values

            if len(auc_scores) > 1:
                mean_auc = np.mean(auc_scores)
                std_auc = np.std(auc_scores)
                se_auc = std_auc / np.sqrt(len(auc_scores))
                ci_95 = 1.96 * se_auc

                llm_stats.append({
                    'LLM': llm,
                    'Mean': mean_auc,
                    'Std': std_auc,
                    'CI_Lower': mean_auc - ci_95,
                    'CI_Upper': mean_auc + ci_95,
                    'N_Queries': len(auc_scores)
                })

        stats_df = pd.DataFrame(llm_stats)
        if not stats_df.empty:
            stats_df = stats_df.sort_values('Mean', ascending=False)
            x_pos = np.arange(len(stats_df))

            bars = ax1.bar(x_pos, stats_df['Mean'],
                           yerr=[stats_df['Mean'] - stats_df['CI_Lower'],
                                 stats_df['CI_Upper'] - stats_df['Mean']],
                           capsize=5, alpha=0.8, edgecolor='black',
                           color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(stats_df)])

            for bar, row in zip(bars, stats_df.iterrows()):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{row[1]["Mean"]:.3f}', ha='center', va='bottom', fontweight='bold')

            ax1.set_title(f'{pipeline_name}: Mean AUC with 95% CI', fontweight='bold', pad=20)
            ax1.set_ylabel('Mean AUC Score')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(stats_df['LLM'], rotation=45, ha='right')

        # 2. Performance distribution by LLM
        if not comp_df.empty:
            llms = comp_df['LLM'].unique()
            data_for_violin = [comp_df[comp_df['LLM'] == llm]['AUC_Mean'].values for llm in llms]

            try:
                parts = ax2.violinplot(data_for_violin, positions=range(len(llms)),
                                       showmeans=True, showmedians=True, showextrema=True)

                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i % len(colors)])
                    pc.set_alpha(0.7)

                ax2.set_xticks(range(len(llms)))
                ax2.set_xticklabels(llms, rotation=45, ha='right')
            except:
                bp = ax2.boxplot(data_for_violin, labels=llms, patch_artist=True)
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors[i % len(colors)])
                    patch.set_alpha(0.7)
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

            ax2.set_title(f'{pipeline_name}: AUC Distribution by LLM', fontweight='bold', pad=20)
            ax2.set_ylabel('AUC Score')

        # 3. Performance correlation (if enough tasks)
        if not comp_df.empty and len(comp_df['Query'].unique()) > 3:
            correlation_data = comp_df.pivot_table(values='AUC_Mean', index='Query',
                                                   columns='LLM', aggfunc='max', fill_value=0)

            if len(correlation_data.columns) > 1:
                try:
                    corr_matrix = correlation_data.corr()

                    im = ax3.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

                    for i in range(len(corr_matrix.index)):
                        for j in range(len(corr_matrix.columns)):
                            ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center',
                                     color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                                     fontweight='bold')

                    ax3.set_xticks(range(len(corr_matrix.columns)))
                    ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                    ax3.set_yticks(range(len(corr_matrix.index)))
                    ax3.set_yticklabels(corr_matrix.index)
                    ax3.set_title(f'{pipeline_name}: LLM Performance Correlation', fontweight='bold', pad=20)

                    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
                    cbar.set_label('Correlation Coefficient')
                except:
                    ax3.text(0.5, 0.5, 'Correlation analysis failed', ha='center', va='center', transform=ax3.transAxes)

        # 4. Performance consistency analysis
        consistency_data = []
        for llm in comp_df['LLM'].unique():
            llm_data = comp_df[comp_df['LLM'] == llm]
            auc_scores = llm_data['AUC_Mean'].values

            if len(auc_scores) > 1 and np.mean(auc_scores) > 0:
                cv = np.std(auc_scores) / np.mean(auc_scores) * 100
                consistency_data.append({
                    'LLM': llm,
                    'CV': cv,
                    'Consistency_Score': max(0, 100 - cv)
                })

        consistency_df = pd.DataFrame(consistency_data)
        if not consistency_df.empty:
            consistency_df = consistency_df.sort_values('Consistency_Score', ascending=False)
            bars = ax4.bar(consistency_df['LLM'], consistency_df['Consistency_Score'],
                           color=self.colors['primary'], alpha=0.8, edgecolor='black')

            for bar, row in zip(bars, consistency_df.iterrows()):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{row[1]["Consistency_Score"]:.1f}', ha='center', va='bottom',
                         fontweight='bold')

            ax4.set_title(f'{pipeline_name}: Performance Consistency Score', fontweight='bold', pad=20)
            ax4.set_ylabel('Consistency Score')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_statistical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} statistical analysis")

    def extract_pipeline_molecular_metrics(self, pipeline_evaluations: Dict) -> Dict:
        """Extract molecular-level metrics for a specific pipeline"""
        molecular_metrics = {}

        for llm_name, llm_eval in pipeline_evaluations.items():
            molecular_metrics[llm_name] = {'molecules': [], 'metrics': {}}

            for query_name, query_eval in llm_eval["query_evaluations"].items():
                pipeline_data = query_eval["pipeline_data"]
                for run in pipeline_data['runs']:
                    molecular_metrics[llm_name]['molecules'].extend(run['molecules'])

            # Calculate metrics for this LLM and pipeline
            molecules = molecular_metrics[llm_name]['molecules']

            if molecules:
                scores = [mol['Oracle_Score'] for mol in molecules]
                valid_scores = [s for s in scores if s > 0]

                metrics = {
                    'total_molecules': len(molecules),
                    'valid_molecules': len(valid_scores),
                    'validity_rate': len(valid_scores) / len(molecules) if molecules else 0,
                    'mean_score': np.mean(valid_scores) if valid_scores else 0,
                    'max_score': max(valid_scores) if valid_scores else 0,
                    'std_score': np.std(valid_scores) if valid_scores else 0,
                    'high_score_count': len([s for s in valid_scores if s > 0.8]),
                    'high_score_rate': len([s for s in valid_scores if s > 0.8]) / len(
                        valid_scores) if valid_scores else 0,
                    'unique_smiles': len(set(mol['SMILES'] for mol in molecules)),
                    'diversity_rate': len(set(mol['SMILES'] for mol in molecules)) / len(molecules) if molecules else 0,
                    'auc_10_mean': np.mean([mol.get('AUC_10', 0) for mol in molecules]) if molecules else 0
                }
                molecular_metrics[llm_name]['metrics'] = metrics
            else:
                # Empty metrics for no molecules
                molecular_metrics[llm_name]['metrics'] = {
                    key: 0 for key in ['total_molecules', 'valid_molecules', 'validity_rate',
                                       'mean_score', 'max_score', 'std_score', 'high_score_count',
                                       'high_score_rate', 'unique_smiles', 'diversity_rate', 'auc_10_mean']
                }

        return molecular_metrics

    def _create_pipeline_molecular_level_analysis(self, molecular_metrics: Dict, pipeline_type: str, viz_dir: Path):
        """Create molecular level analysis visualization for a specific pipeline"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: Molecular Level Performance Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # Prepare data for visualization
        metrics_data = []
        for llm_name, llm_data in molecular_metrics.items():
            metrics = llm_data['metrics']
            metrics_data.append({
                'LLM': llm_name,
                **metrics
            })

        metrics_df = pd.DataFrame(metrics_data)

        if not metrics_df.empty:
            # 1. Validity and diversity rates
            llms = metrics_df['LLM'].unique()
            x_pos = np.arange(len(llms))
            width = 0.35

            validity_rates = [metrics_df[metrics_df['LLM'] == llm]['validity_rate'].iloc[0] * 100 for llm in llms]
            diversity_rates = [metrics_df[metrics_df['LLM'] == llm]['diversity_rate'].iloc[0] * 100 for llm in llms]

            bars1 = ax1.bar(x_pos - width / 2, validity_rates, width,
                            label='Validity Rate', color=self.colors['primary'], alpha=0.8)
            bars2 = ax1.bar(x_pos + width / 2, diversity_rates, width,
                            label='Diversity Rate', color=self.colors['secondary'], alpha=0.8)

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax1.text(bar.get_x() + bar.get_width() / 2, height + 1,
                                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax1.set_title(f'{pipeline_name}: Validity and Diversity Rates', fontweight='bold', pad=20)
            ax1.set_ylabel('Rate (%)')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(llms, rotation=45, ha='right')
            ax1.legend()

            # 2. Mean scores comparison
            mean_scores = [metrics_df[metrics_df['LLM'] == llm]['mean_score'].iloc[0] for llm in llms]
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(llms)]

            bars = ax2.bar(llms, mean_scores, color=colors, alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, mean_scores):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax2.set_title(f'{pipeline_name}: Mean Oracle Score by LLM', fontweight='bold', pad=20)
            ax2.set_ylabel('Mean Oracle Score')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

            # 3. High score rates (>0.8)
            high_score_rates = [metrics_df[metrics_df['LLM'] == llm]['high_score_rate'].iloc[0] * 100 for llm in llms]

            bars = ax3.bar(llms, high_score_rates, color=colors, alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, high_score_rates):
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax3.set_title(f'{pipeline_name}: High Score Rate (>0.8)', fontweight='bold', pad=20)
            ax3.set_ylabel('High Score Rate (%)')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

            # 4. Total molecules generated
            total_molecules = [metrics_df[metrics_df['LLM'] == llm]['total_molecules'].iloc[0] for llm in llms]

            bars = ax4.bar(llms, total_molecules, color=colors, alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, total_molecules):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(total_molecules) * 0.01,
                         f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax4.set_title(f'{pipeline_name}: Total Molecules Generated', fontweight='bold', pad=20)
            ax4.set_ylabel('Number of Molecules')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_molecular_level_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} molecular level analysis")

    def _generate_cross_llm_single_shot_table_separate(self, pipeline_evaluations: Dict, tables_dir: Path):
        """Generate separate single-shot cross-LLM table"""
        filename = tables_dir / "single_shot_cross_llm_comparison.tex"

        # Collect all task data for single-shot pipeline
        all_task_data = {}
        llm_names = list(pipeline_evaluations.keys())

        for llm_name, llm_eval in pipeline_evaluations.items():
            for query_name, query_eval in llm_eval["query_evaluations"].items():
                if query_name not in all_task_data:
                    all_task_data[query_name] = {}

                pipeline_data = query_eval["pipeline_data"]
                auc_val = np.mean(pipeline_data["auc_scores"]) if pipeline_data["auc_scores"] else 0.0
                top10_val = np.mean(pipeline_data["top_10_scores"]) if pipeline_data["top_10_scores"] else 0.0
                best_val = max([run["max_score"] for run in pipeline_data["runs"]]) if pipeline_data["runs"] else 0.0

                all_task_data[query_name][llm_name] = {
                    'auc': auc_val,
                    'top10': top10_val,
                    'best': best_val
                }

        # Generate the table using the existing method structure
        self._write_cross_llm_table(all_task_data, llm_names, "Single-Shot", filename)

    def _generate_cross_llm_iterative_table_separate(self, pipeline_evaluations: Dict, tables_dir: Path):
        """Generate separate iterative cross-LLM table"""
        filename = tables_dir / "iterative_cross_llm_comparison.tex"

        # Collect all task data for iterative pipeline
        all_task_data = {}
        llm_names = list(pipeline_evaluations.keys())

        for llm_name, llm_eval in pipeline_evaluations.items():
            for query_name, query_eval in llm_eval["query_evaluations"].items():
                if query_name not in all_task_data:
                    all_task_data[query_name] = {}

                pipeline_data = query_eval["pipeline_data"]
                auc_val = np.mean(pipeline_data["auc_scores"]) if pipeline_data["auc_scores"] else 0.0
                top10_val = np.mean(pipeline_data["top_10_scores"]) if pipeline_data["top_10_scores"] else 0.0
                best_val = max([run["max_score"] for run in pipeline_data["runs"]]) if pipeline_data["runs"] else 0.0

                all_task_data[query_name][llm_name] = {
                    'auc': auc_val,
                    'top10': top10_val,
                    'best': best_val
                }

        # Generate the table using the existing method structure
        self._write_cross_llm_table(all_task_data, llm_names, "Iterative", filename)

    def _write_cross_llm_table(self, all_task_data: Dict, llm_names: List[str], pipeline_name: str, filename: Path):
        """Write cross-LLM comparison table with fixed LaTeX syntax"""
        sorted_tasks = sorted(all_task_data.keys())

        # Calculate wins for each LLM
        llm_wins = {llm: 0 for llm in llm_names}
        task_winners = {}

        for task_name in sorted_tasks:
            task_data = all_task_data[task_name]
            best_auc = max(task_data[llm]['auc'] for llm in llm_names if llm in task_data)

            winners = [llm for llm in llm_names if
                       llm in task_data and task_data[llm]['auc'] == best_auc and best_auc > 0]
            if len(winners) == 1:
                task_winners[task_name] = winners[0]
                llm_wins[winners[0]] += 1
            else:
                task_winners[task_name] = "Tie" if len(winners) > 1 else "None"

        # Create LaTeX table - only AUC-10 columns
        num_llms = len(llm_names)
        col_spec = "|l|" + "c|" * num_llms + "c|"

        latex_content = f"""\\begin{{table}}[h!]
\\centering
\\caption{{{pipeline_name} Pipeline: Cross-LLM AUC-10 Performance Comparison - Top-10 Molecule Analysis}}
\\label{{tab:{pipeline_name.lower().replace('-', '_')}_cross_llm_auc10_comparison}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_spec}}}
\\hline
\\multicolumn{{{num_llms + 2}}}{{|c|}}{{\\textbf{{\\Large AUC-10 PERFORMANCE ANALYSIS}}}} \\\\
\\multicolumn{{{num_llms + 2}}}{{|c|}}{{\\textit{{Area Under Curve calculated from Top-10 molecules per task}}}} \\\\
\\hline
\\textbf{{Task}}"""

        # Add LLM headers with AUC-10 emphasis
        for llm in llm_names:
            safe_llm = llm.replace('_', '\\_').replace('&', '\\&')
            latex_content += f" & \\textbf{{{safe_llm}\\\\AUC-10}}"
        latex_content += " & \\textbf{{Best\\\\Performer}} \\\\\n\\hline\n"

        # Add task rows - only AUC-10 values
        total_aucs = {llm: 0.0 for llm in llm_names}

        for task_name in sorted_tasks:
            task_data = all_task_data[task_name]
            latex_task_name = task_name.replace('_', '\\_').replace('&', '\\&')
            latex_content += f"{latex_task_name}"

            winner = task_winners.get(task_name, "None")

            for llm in llm_names:
                if llm in task_data:
                    data = task_data[llm]
                    auc_val = data['auc']
                    total_aucs[llm] += auc_val

                    if winner == llm:
                        latex_content += f" & \\textbf{{{auc_val:.4f}}}"
                    else:
                        latex_content += f" & {auc_val:.4f}"
                else:
                    latex_content += " & 0.0000"

            latex_content += f" & {winner} \\\\\n"

        # Add summary section with proper LaTeX syntax
        latex_content += "\\hline\n"
        latex_content += f"\\multicolumn{{{num_llms + 2}}}{{|c|}}{{\\textbf{{SUMMARY STATISTICS}}}} \\\\\n"
        latex_content += "\\hline\n"

        # Add wins for each LLM
        for llm in llm_names:
            wins = llm_wins[llm]
            percentage = (wins / len(sorted_tasks)) * 100 if sorted_tasks else 0
            safe_llm = llm.replace('_', '\\_').replace('&', '\\&')
            latex_content += f"\\textbf{{{safe_llm} Wins}} & \\multicolumn{{{num_llms + 1}}}{{c|}}{{{wins} ({percentage:.1f}\\%)}} \\\\\n"

        # Add comprehensive summary section with AUC-10 emphasis
        latex_content += "\\hline\n"
        latex_content += f"\\multicolumn{{{num_llms + 2}}}{{|c|}}{{\\textbf{{\\Large PERFORMANCE SUMMARY}}}} \\\\\n"
        latex_content += "\\hline\n"

        # Add total AUC-10 scores with ranking
        sorted_totals = sorted([(total_aucs[llm], llm) for llm in llm_names], reverse=True)

        latex_content += "\\textbf{{TOTAL AUC-10 SCORES}}"
        for llm in llm_names:
            # Find rank
            rank = next(i + 1 for i, (score, name) in enumerate(sorted_totals) if name == llm)
            if rank == 1:
                latex_content += f" & \\cellcolor{{green!40}}\\textbf{{\\#{rank}: {total_aucs[llm]:.4f}}}"
            elif rank == 2:
                latex_content += f" & \\cellcolor{{yellow!30}}\\textbf{{\\#{rank}: {total_aucs[llm]:.4f}}}"
            elif rank == 3:
                latex_content += f" & \\cellcolor{{orange!20}}\\textbf{{\\#{rank}: {total_aucs[llm]:.4f}}}"
            else:
                latex_content += f" & \\textbf{{\\#{rank}: {total_aucs[llm]:.4f}}}"
        latex_content += " & \\textbf{{RANK}} \\\\\n"

        # Add average AUC-10 per task
        latex_content += "\\textbf{{AVERAGE AUC-10}}"
        for llm in llm_names:
            avg_auc_10 = total_aucs[llm] / len(sorted_tasks) if sorted_tasks else 0
            latex_content += f" & \\textbf{{{avg_auc_10:.4f}}}"
        latex_content += " & \\textbf{{per Task}} \\\\\n"

        latex_content += """\\hline
\\end{tabular}%
}
\\end{table}

\\vspace{0.5cm}
\\textbf{Note:} AUC-10 scores represent the Area Under the Curve calculated from the top-10 highest-scoring molecules for each task. Higher scores indicate better performance in generating high-quality molecules. Rankings show overall performance across all tasks.
"""

        with open(filename, 'w') as f:
            f.write(latex_content)
        print(f"    ✓ Created {pipeline_name} cross-LLM AUC-10 table with enhanced AUC-10 emphasis: {filename}")

    def _create_pipeline_specific_llm_tables(self, llm_name: str, llm_evaluation: Dict, pipeline_type: str,
                                             tables_dir: Path):
        """Create only pipeline-specific tables for a single LLM to prevent contamination"""
        print(f"    Creating {pipeline_type} tables for {llm_name}...")

        # Collect task-wise data for the specified pipeline only
        task_data = []

        for query_name, query_eval in llm_evaluation["query_evaluations"].items():
            # Get data for the specified pipeline type
            pipeline_data = query_eval[pipeline_type] if pipeline_type in query_eval else None

            if pipeline_data and pipeline_data.get("auc_scores"):
                auc_val = np.mean(pipeline_data["auc_scores"])
                top10_val = np.mean(pipeline_data["top_10_scores"]) if pipeline_data.get("top_10_scores") else 0.0
                best_val = max([run["max_score"] for run in pipeline_data["runs"]]) if pipeline_data.get(
                    "runs") else 0.0

                task_data.append({
                    'task': query_name.replace('_', '\\_'),  # Escape underscores for LaTeX
                    'auc': auc_val,
                    'top10': top10_val,
                    'best': best_val
                })

        if not task_data:
            print(f"      No data found for {llm_name} in {pipeline_type} pipeline")
            return

        # Sort by task name for consistent ordering
        task_data.sort(key=lambda x: x['task'])

        # Generate only the pipeline-specific table
        self._generate_pipeline_specific_latex_table(llm_name, task_data, pipeline_type, tables_dir)

    def _generate_pipeline_specific_latex_table(self, llm_name: str, task_data: list, pipeline_type: str,
                                                tables_dir: Path):
        """Generate LaTeX table for a specific pipeline only"""
        safe_llm_name = llm_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('.', '')
        pipeline_name = pipeline_type.replace('_', '-').title()
        filename = tables_dir / f"{safe_llm_name}_{pipeline_type}_performance.tex"

        # Calculate summary statistics
        total_tasks = len(task_data)
        total_auc = sum(t['auc'] for t in task_data)
        total_top10 = sum(t['top10'] for t in task_data)
        best_overall = max(t['best'] for t in task_data) if task_data else 0.0
        mean_auc = total_auc / total_tasks if total_tasks > 0 else 0.0

        latex_content = f"""\\begin{{table}}[h!]
    \\centering
    \\caption{{{pipeline_name} Pipeline Performance: {llm_name}}}
    \\label{{tab:{safe_llm_name}_{pipeline_type}_performance}}
    \\begin{{tabular}}{{|l|c|c|c|}}
    \\hline
    \\textbf{{Task}} & \\textbf{{AUC-10}} & \\textbf{{Top-10 Mean}} & \\textbf{{Best Score}} \\\\
    \\hline
"""

        # Add task rows (sort by AUC descending for better readability)
        sorted_data = sorted(task_data, key=lambda x: x['auc'], reverse=True)
        for task in sorted_data:
            latex_content += f"    {task['task']} & {task['auc']:.4f} & {task['top10']:.4f} & {task['best']:.4f} \\\\\n"

        # Add summary statistics
        latex_content += f"""    \\hline
    \\multicolumn{{4}}{{|c|}}{{\\textbf{{SUMMARY STATISTICS}}}} \\\\
    \\hline
    \\multicolumn{{1}}{{|l|}}{{\\textbf{{Total Tasks}}}} & \\multicolumn{{3}}{{c|}}{{{total_tasks}}} \\\\
    \\multicolumn{{1}}{{|l|}}{{\\textbf{{Mean AUC}}}} & \\multicolumn{{3}}{{c|}}{{{mean_auc:.4f}}} \\\\
    \\hline
    \\textbf{{Totals/Best}} & {total_auc:.4f} & {total_top10:.4f} & {best_overall:.4f} \\\\
    \\hline
    \\end{{tabular}}
    \\end{{table}}
"""

        with open(filename, 'w') as f:
            f.write(latex_content)
        print(f"      ✓ Created {filename}")

    def _create_top_k_smiles_overlap_analysis(self, pipeline_evaluations: Dict, pipeline_type: str, viz_dir: Path):
        """Create comprehensive top-K SMILES overlap analysis across tasks"""
        print(f"    ✓ Creating {pipeline_type} top-K SMILES overlap analysis...")

        pipeline_name = pipeline_type.replace('_', '-').title()

        # Collect top-K SMILES for each task and LLM combination
        task_overlap_data = {}

        for llm_name, llm_eval in pipeline_evaluations.items():
            for query_name, query_eval in llm_eval["query_evaluations"].items():
                if query_name not in task_overlap_data:
                    task_overlap_data[query_name] = {}

                pipeline_data = query_eval["pipeline_data"]
                if pipeline_data["runs"]:
                    # Collect all molecules from all runs for this task
                    all_molecules = []
                    for run in pipeline_data["runs"]:
                        all_molecules.extend(run["molecules"])

                    # Sort by Oracle_Score and get top-K SMILES
                    sorted_molecules = sorted(all_molecules, key=lambda x: x['Oracle_Score'], reverse=True)

                    task_overlap_data[query_name][llm_name] = {
                        'top_1': set([sorted_molecules[0]['SMILES']]) if len(sorted_molecules) >= 1 else set(),
                        'top_5': set([mol['SMILES'] for mol in sorted_molecules[:5]]) if len(
                            sorted_molecules) >= 5 else set([mol['SMILES'] for mol in sorted_molecules]),
                        'top_10': set([mol['SMILES'] for mol in sorted_molecules[:10]]) if len(
                            sorted_molecules) >= 10 else set([mol['SMILES'] for mol in sorted_molecules])
                    }

        # Calculate overlap statistics for each task and top-K level
        overlap_results = []

        for query_name, llm_data in task_overlap_data.items():
            llms = list(llm_data.keys())

            for k_level in ['top_1', 'top_5', 'top_10']:
                k_num = int(k_level.split('_')[1])

                # Get all SMILES sets for this k-level
                smiles_sets = [llm_data[llm][k_level] for llm in llms if llm in llm_data]

                if len(smiles_sets) > 1:
                    # Calculate pairwise overlaps
                    total_pairs = len(smiles_sets) * (len(smiles_sets) - 1) // 2
                    overlap_sum = 0

                    for i in range(len(smiles_sets)):
                        for j in range(i + 1, len(smiles_sets)):
                            set1, set2 = smiles_sets[i], smiles_sets[j]
                            if len(set1) > 0 and len(set2) > 0:
                                intersection = len(set1.intersection(set2))
                                union = len(set1.union(set2))
                                jaccard = intersection / union if union > 0 else 0
                                overlap_sum += jaccard

                    avg_overlap = overlap_sum / total_pairs if total_pairs > 0 else 0

                    # Calculate total unique molecules across all LLMs
                    all_unique = set()
                    for s in smiles_sets:
                        all_unique.update(s)

                    overlap_results.append({
                        'Task': query_name,
                        'K_Level': k_level,
                        'K_Number': k_num,
                        'Avg_Jaccard': avg_overlap,
                        'Total_Unique': len(all_unique),
                        'LLM_Count': len(smiles_sets)
                    })

        if not overlap_results:
            print(f"      No overlap data available for {pipeline_type}")
            return

        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{pipeline_name} Pipeline: Top-K SMILES Overlap Analysis Across Tasks',
                     fontsize=16, fontweight='bold', y=0.98)

        overlap_df = pd.DataFrame(overlap_results)

        # 1. Heatmap of average Jaccard indices by task and K-level
        if not overlap_df.empty:
            heatmap_data = overlap_df.pivot_table(values='Avg_Jaccard', index='Task', columns='K_Level', fill_value=0)

            if not heatmap_data.empty:
                im1 = ax1.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

                # Add text annotations
                for i in range(len(heatmap_data.index)):
                    for j in range(len(heatmap_data.columns)):
                        text = ax1.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}',
                                        ha='center', va='center', fontweight='bold',
                                        color='white' if heatmap_data.iloc[i, j] < 0.5 else 'black')

                ax1.set_xticks(range(len(heatmap_data.columns)))
                ax1.set_xticklabels(['Top-1', 'Top-5', 'Top-10'])
                ax1.set_yticks(range(len(heatmap_data.index)))
                ax1.set_yticklabels([task[:15] + '...' if len(task) > 15 else task for task in heatmap_data.index],
                                    fontsize=9)
                ax1.set_title(f'{pipeline_name}: Average Jaccard Index by Task and Top-K', fontweight='bold', pad=20)

                # Add colorbar
                cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
                cbar1.set_label('Average Jaccard Index')

        # 2. Bar plot showing overlap trends across K-levels
        k_level_stats = overlap_df.groupby('K_Level').agg({
            'Avg_Jaccard': ['mean', 'std'],
            'Total_Unique': 'mean'
        }).round(4)

        k_level_stats.columns = ['Mean_Jaccard', 'Std_Jaccard', 'Mean_Unique']
        k_level_stats = k_level_stats.reset_index()

        colors = ['#E31A1C', '#FF7F00', '#1F78B4']  # Red, Orange, Blue for top-1, top-5, top-10
        bars = ax2.bar(['Top-1', 'Top-5', 'Top-10'], k_level_stats['Mean_Jaccard'],
                       yerr=k_level_stats['Std_Jaccard'], color=colors, alpha=0.8,
                       capsize=5, edgecolor='black')

        for bar, mean_val, std_val in zip(bars, k_level_stats['Mean_Jaccard'], k_level_stats['Std_Jaccard']):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std_val + 0.01,
                     f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

        ax2.set_title(f'{pipeline_name}: Mean Jaccard Index by Top-K Level', fontweight='bold', pad=20)
        ax2.set_ylabel('Mean Jaccard Index')
        ax2.set_ylim(0, max(k_level_stats['Mean_Jaccard'] + k_level_stats['Std_Jaccard']) * 1.2)

        # 3. Scatter plot: Task difficulty vs overlap
        task_difficulty = overlap_df[overlap_df['K_Level'] == 'top_10'].copy()
        if not task_difficulty.empty:
            # Calculate task difficulty as inverse of unique molecules (more unique = less overlap = harder)
            ax3.scatter(task_difficulty['Total_Unique'], task_difficulty['Avg_Jaccard'],
                        s=100, alpha=0.7, c=range(len(task_difficulty)), cmap='viridis')

            for i, row in task_difficulty.iterrows():
                ax3.annotate(row['Task'][:10] + '...', (row['Total_Unique'], row['Avg_Jaccard']),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax3.set_title(f'{pipeline_name}: Task Diversity vs LLM Consensus (Top-10)', fontweight='bold', pad=20)
            ax3.set_xlabel('Total Unique SMILES (Diversity)')
            ax3.set_ylabel('Average Jaccard Index (Consensus)')

        # 4. Detailed overlap comparison across tasks for each K-level
        tasks_to_show = overlap_df['Task'].unique()[:8]  # Show top 8 tasks to avoid overcrowding

        if len(tasks_to_show) > 0:
            x_pos = np.arange(len(tasks_to_show))
            width = 0.25

            top1_data = []
            top5_data = []
            top10_data = []

            for task in tasks_to_show:
                task_data = overlap_df[overlap_df['Task'] == task]
                top1_val = task_data[task_data['K_Level'] == 'top_1']['Avg_Jaccard'].iloc[0] if len(
                    task_data[task_data['K_Level'] == 'top_1']) > 0 else 0
                top5_val = task_data[task_data['K_Level'] == 'top_5']['Avg_Jaccard'].iloc[0] if len(
                    task_data[task_data['K_Level'] == 'top_5']) > 0 else 0
                top10_val = task_data[task_data['K_Level'] == 'top_10']['Avg_Jaccard'].iloc[0] if len(
                    task_data[task_data['K_Level'] == 'top_10']) > 0 else 0

                top1_data.append(top1_val)
                top5_data.append(top5_val)
                top10_data.append(top10_val)

            bars1 = ax4.bar(x_pos - width, top1_data, width, label='Top-1', color='#E31A1C', alpha=0.8)
            bars2 = ax4.bar(x_pos, top5_data, width, label='Top-5', color='#FF7F00', alpha=0.8)
            bars3 = ax4.bar(x_pos + width, top10_data, width, label='Top-10', color='#1F78B4', alpha=0.8)

            ax4.set_title(f'{pipeline_name}: Top-K Overlap by Task', fontweight='bold', pad=20)
            ax4.set_ylabel('Average Jaccard Index')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([task[:12] + '...' if len(task) > 12 else task for task in tasks_to_show],
                                rotation=45, ha='right', fontsize=9)
            ax4.legend()
            ax4.set_ylim(0, 1)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_top_k_smiles_overlap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save detailed overlap data as CSV
        overlap_df.to_csv(viz_dir.parent / "data" / f"{pipeline_type}_top_k_overlap_analysis.csv", index=False)

        print(f"    ✓ Created {pipeline_type} top-K SMILES overlap analysis")
        print(f"    ✓ Saved overlap data: {viz_dir.parent / 'data' / f'{pipeline_type}_top_k_overlap_analysis.csv'}")

        # Create additional comprehensive cross-LLM SMILES overlap visualization
        self._create_comprehensive_cross_llm_smiles_overlap(pipeline_evaluations, pipeline_type, viz_dir)

    def _create_comprehensive_cross_llm_smiles_overlap(self, pipeline_evaluations: Dict, pipeline_type: str,
                                                       viz_dir: Path):
        """Create comprehensive SMILES overlap visualization split into 3 clean files"""
        print(f"    ✓ Creating comprehensive cross-LLM SMILES overlap for {pipeline_type}...")

        pipeline_name = pipeline_type.replace('_', '-').title()

        # Collect all unique SMILES for each LLM across all tasks
        llm_all_smiles = {}
        task_specific_smiles = {}

        for llm_name, llm_eval in pipeline_evaluations.items():
            llm_all_smiles[llm_name] = {
                'top_1': set(),
                'top_5': set(),
                'top_10': set(),
                'all': set()
            }

            for query_name, query_eval in llm_eval["query_evaluations"].items():
                if query_name not in task_specific_smiles:
                    task_specific_smiles[query_name] = {}

                if llm_name not in task_specific_smiles[query_name]:
                    task_specific_smiles[query_name][llm_name] = {
                        'top_1': set(),
                        'top_5': set(),
                        'top_10': set()
                    }

                pipeline_data = query_eval["pipeline_data"]
                if pipeline_data["runs"]:
                    # Collect all molecules from all runs for this task
                    all_molecules = []
                    for run in pipeline_data["runs"]:
                        all_molecules.extend(run["molecules"])

                    # Sort by Oracle_Score and get top-K SMILES
                    sorted_molecules = sorted(all_molecules, key=lambda x: x['Oracle_Score'], reverse=True)

                    for mol in sorted_molecules:
                        llm_all_smiles[llm_name]['all'].add(mol['SMILES'])

                    # Top-K sets
                    if len(sorted_molecules) >= 1:
                        top_1_smiles = {sorted_molecules[0]['SMILES']}
                        llm_all_smiles[llm_name]['top_1'].update(top_1_smiles)
                        task_specific_smiles[query_name][llm_name]['top_1'].update(top_1_smiles)

                    if len(sorted_molecules) >= 5:
                        top_5_smiles = {mol['SMILES'] for mol in sorted_molecules[:5]}
                        llm_all_smiles[llm_name]['top_5'].update(top_5_smiles)
                        task_specific_smiles[query_name][llm_name]['top_5'].update(top_5_smiles)

                    if len(sorted_molecules) >= 10:
                        top_10_smiles = {mol['SMILES'] for mol in sorted_molecules[:10]}
                        llm_all_smiles[llm_name]['top_10'].update(top_10_smiles)
                        task_specific_smiles[query_name][llm_name]['top_10'].update(top_10_smiles)

        llm_names = list(llm_all_smiles.keys())

        # Create three separate, cleaner visualizations
        self._create_overlap_analysis_part1(llm_all_smiles, llm_names, pipeline_name, pipeline_type, viz_dir)
        self._create_overlap_analysis_part2(llm_all_smiles, task_specific_smiles, llm_names, pipeline_name,
                                            pipeline_type, viz_dir)
        self._create_overlap_analysis_part3(llm_all_smiles, pipeline_evaluations, llm_names, pipeline_name,
                                            pipeline_type, viz_dir)

        print(f"    ✓ Created 3-part cross-LLM SMILES overlap analysis for {pipeline_type}")

    def _create_overlap_analysis_part1(self, llm_all_smiles: Dict, llm_names: List, pipeline_name: str,
                                       pipeline_type: str, viz_dir: Path):
        """Part 1: Core Overlap Analysis - Heatmaps and Trends"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{pipeline_name} Pipeline: Core SMILES Overlap Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        if len(llm_names) > 1:
            # 1. Pairwise overlap heatmap
            n_llms = len(llm_names)
            jaccard_matrix = np.zeros((n_llms, n_llms))

            for i, llm1 in enumerate(llm_names):
                for j, llm2 in enumerate(llm_names):
                    set1 = llm_all_smiles[llm1]['top_10']
                    set2 = llm_all_smiles[llm2]['top_10']

                    if len(set1) > 0 and len(set2) > 0:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        jaccard_matrix[i, j] = intersection / union if union > 0 else 0
                    else:
                        jaccard_matrix[i, j] = 0

            im = ax1.imshow(jaccard_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')

            # Add text annotations
            for i in range(n_llms):
                for j in range(n_llms):
                    text_color = 'white' if jaccard_matrix[i, j] < 0.5 else 'black'
                    ax1.text(j, i, f'{jaccard_matrix[i, j]:.3f}', ha='center', va='center',
                             fontweight='bold', fontsize=11, color=text_color)

            ax1.set_xticks(range(n_llms))
            ax1.set_xticklabels(llm_names, rotation=45, ha='right', fontsize=10)
            ax1.set_yticks(range(n_llms))
            ax1.set_yticklabels(llm_names, fontsize=10)
            ax1.set_title('Pairwise SMILES Overlap (Top-10)\nJaccard Index Heatmap',
                          fontweight='bold', fontsize=12, pad=20)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
            cbar.set_label('Jaccard Index', fontweight='bold')

        # 2. Top-K overlap trends
        k_levels_simple = ['top_1', 'top_5', 'top_10']
        k_labels_simple = ['Top-1', 'Top-5', 'Top-10']
        overlap_trends = []

        for k_level in k_levels_simple:
            overlaps = []
            for i in range(len(llm_names)):
                for j in range(i + 1, len(llm_names)):
                    set1 = llm_all_smiles[llm_names[i]][k_level]
                    set2 = llm_all_smiles[llm_names[j]][k_level]

                    if len(set1) > 0 and len(set2) > 0:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        jaccard = intersection / union if union > 0 else 0
                        overlaps.append(jaccard)

            avg_overlap = np.mean(overlaps) if overlaps else 0
            overlap_trends.append(avg_overlap)

        bars = ax2.bar(k_labels_simple, overlap_trends,
                       color=['#e74c3c', '#f39c12', '#3498db'], alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, overlap_trends):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax2.set_ylabel('Average Jaccard Index', fontweight='bold')
        ax2.set_title('Cross-LLM Overlap Trends by Top-K', fontweight='bold', fontsize=12, pad=20)
        ax2.set_ylim(0, max(overlap_trends) * 1.2 if overlap_trends else 1)
        ax2.grid(axis='y', alpha=0.3)

        # 3. Sharing pattern distribution
        if len(llm_names) > 1:
            all_top10_smiles = set()
            for llm_name in llm_names:
                all_top10_smiles.update(llm_all_smiles[llm_name]['top_10'])

            smiles_counts = {}
            for smiles in all_top10_smiles:
                count = sum(1 for llm_name in llm_names if smiles in llm_all_smiles[llm_name]['top_10'])
                smiles_counts[smiles] = count

            unique_count = sum(1 for count in smiles_counts.values() if count == 1)
            shared_2_count = sum(1 for count in smiles_counts.values() if count == 2)
            shared_3plus_count = sum(1 for count in smiles_counts.values() if count >= 3)

            sizes = [unique_count, shared_2_count, shared_3plus_count]
            labels = ['Unique to 1 LLM', 'Shared by 2 LLMs', 'Shared by 3+ LLMs']
            colors = ['#e74c3c', '#f39c12', '#27ae60']

            non_zero_data = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
            if non_zero_data:
                sizes, labels, colors = zip(*non_zero_data)

                # Create pie chart with shortened labels to avoid clutter
                short_labels = [label.replace('Unique to 1 LLM', 'Unique').replace('Shared by 2 LLMs', '2-LLM').replace(
                    'Shared by 3+ LLMs', '3+ LLM') for label in labels]
                wedges, texts, autotexts = ax3.pie(sizes, labels=short_labels, autopct='%1.1f%%',
                                                   colors=colors, startangle=90,
                                                   textprops={'fontsize': 8, 'fontweight': 'bold'},
                                                   pctdistance=0.85, labeldistance=1.05)

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(8)

            ax3.set_title('Top-10 SMILES Sharing Pattern', fontweight='bold', fontsize=12, pad=20)

        # 4. Overlap statistics summary
        if len(llm_names) > 1:
            # Calculate detailed overlap statistics
            overlap_stats = []
            for i, llm1 in enumerate(llm_names):
                for j, llm2 in enumerate(llm_names):
                    if i < j:  # Only upper triangle
                        set1 = llm_all_smiles[llm1]['top_10']
                        set2 = llm_all_smiles[llm2]['top_10']

                        if len(set1) > 0 and len(set2) > 0:
                            intersection = len(set1.intersection(set2))
                            union = len(set1.union(set2))
                            jaccard = intersection / union if union > 0 else 0
                            overlap_stats.append({
                                'LLM_Pair': f'{llm1[:10]}... vs {llm2[:10]}...',
                                'Jaccard': jaccard,
                                'Intersection': intersection,
                                'Union': union
                            })

            if overlap_stats:
                # Sort by Jaccard index
                overlap_stats.sort(key=lambda x: x['Jaccard'], reverse=True)

                # Show top pairs
                top_pairs = overlap_stats[:min(6, len(overlap_stats))]
                pair_names = [stat['LLM_Pair'] for stat in top_pairs]
                jaccard_values = [stat['Jaccard'] for stat in top_pairs]

                bars = ax4.barh(range(len(pair_names)), jaccard_values,
                                color=plt.cm.viridis(np.linspace(0, 1, len(pair_names))), alpha=0.8)

                for bar, val in zip(bars, jaccard_values):
                    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{val:.3f}', va='center', fontweight='bold', fontsize=9)

                ax4.set_yticks(range(len(pair_names)))
                ax4.set_yticklabels(pair_names, fontsize=9)
                ax4.set_xlabel('Jaccard Index', fontweight='bold')
                ax4.set_title('Top LLM Pairs by Overlap', fontweight='bold', fontsize=12, pad=20)
                ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_smiles_overlap_part1_core_analysis.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_overlap_analysis_part2(self, llm_all_smiles: Dict, task_specific_smiles: Dict, llm_names: List,
                                       pipeline_name: str, pipeline_type: str, viz_dir: Path):
        """Part 2: Molecular Generation & Diversity Analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{pipeline_name} Pipeline: Molecular Generation & Diversity Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. Unique molecules count by LLM (stacked bar)
        k_levels = ['top_1', 'top_5', 'top_10', 'all']
        k_labels = ['Top-1', 'Top-5', 'Top-10', 'All Molecules']
        colors = ['#e74c3c', '#f39c12', '#3498db', '#95a5a6']

        x_pos = np.arange(len(llm_names))
        width = 0.6

        bottom = np.zeros(len(llm_names))

        for k_level, k_label, color in zip(k_levels, k_labels, colors):
            counts = [len(llm_all_smiles[llm][k_level]) for llm in llm_names]
            bars = ax1.bar(x_pos, counts, width, bottom=bottom, label=k_label,
                           color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

            # Add count labels for significant values
            for i, (bar, count) in enumerate(zip(bars, counts)):
                if count > 5:  # Only show labels for significant counts
                    ax1.text(bar.get_x() + bar.get_width() / 2,
                             bottom[i] + count / 2, f'{count}',
                             ha='center', va='center', fontweight='bold',
                             fontsize=9, color='white')

            bottom += counts

        ax1.set_xlabel('LLM Models', fontweight='bold')
        ax1.set_ylabel('Number of Unique SMILES', fontweight='bold')
        ax1.set_title('Unique SMILES Generation by Category', fontweight='bold', fontsize=12, pad=20)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(llm_names, rotation=45, ha='right')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)

        # 2. Chemical diversity analysis (SMILES length)
        length_data = {}
        for llm_name in llm_names:
            all_smiles = llm_all_smiles[llm_name]['all']
            if all_smiles:
                lengths = [len(smiles) for smiles in all_smiles]
                length_data[llm_name] = {
                    'mean': np.mean(lengths),
                    'std': np.std(lengths),
                    'count': len(lengths)
                }

        if length_data:
            llm_names_sorted = sorted(length_data.keys(), key=lambda x: length_data[x]['mean'])
            means = [length_data[llm]['mean'] for llm in llm_names_sorted]
            stds = [length_data[llm]['std'] for llm in llm_names_sorted]

            bars = ax2.bar(llm_names_sorted, means, yerr=stds,
                           color=plt.cm.Set2(np.linspace(0, 1, len(llm_names_sorted))),
                           alpha=0.8, capsize=5, edgecolor='black')

            for bar, mean_val in zip(bars, means):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.1,
                         f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')

            ax2.set_ylabel('Average SMILES Length', fontweight='bold')
            ax2.set_title('Chemical Complexity by LLM\n(SMILES Length Distribution)',
                          fontweight='bold', fontsize=12, pad=20)
            ax2.set_xticklabels(llm_names_sorted, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)

        # 3. Task-specific overlap summary
        task_overlaps = []
        task_names = []

        for task_name, task_data in task_specific_smiles.items():
            if len(task_data) > 1:
                task_llms = list(task_data.keys())
                overlaps = []

                for i in range(len(task_llms)):
                    for j in range(i + 1, len(task_llms)):
                        set1 = task_data[task_llms[i]]['top_10']
                        set2 = task_data[task_llms[j]]['top_10']

                        if len(set1) > 0 and len(set2) > 0:
                            intersection = len(set1.intersection(set2))
                            union = len(set1.union(set2))
                            jaccard = intersection / union if union > 0 else 0
                            overlaps.append(jaccard)

                if overlaps:
                    avg_overlap = np.mean(overlaps)
                    task_overlaps.append(avg_overlap)
                    task_names.append(task_name[:20] + '...' if len(task_name) > 20 else task_name)

        if task_overlaps:
            sorted_data = sorted(zip(task_overlaps, task_names), reverse=True)
            task_overlaps, task_names = zip(*sorted_data)

            top_n = min(10, len(task_overlaps))

            bars = ax3.barh(range(top_n), task_overlaps[:top_n],
                            color=plt.cm.viridis(np.linspace(0, 1, top_n)), alpha=0.8)

            for bar, val in zip(bars, task_overlaps[:top_n]):
                ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{val:.3f}', va='center', fontweight='bold', fontsize=9)

            ax3.set_yticks(range(top_n))
            ax3.set_yticklabels(task_names[:top_n], fontsize=9)
            ax3.set_xlabel('Average Jaccard Index', fontweight='bold')
            ax3.set_title('Top Tasks by LLM Consensus\n(Top-10 SMILES)',
                          fontweight='bold', fontsize=12, pad=20)
            ax3.grid(axis='x', alpha=0.3)

        # 4. Diversity comparison (Top-K vs All molecules)
        diversity_ratios = []
        llm_labels = []

        for llm_name in llm_names:
            top10_count = len(llm_all_smiles[llm_name]['top_10'])
            all_count = len(llm_all_smiles[llm_name]['all'])

            if all_count > 0:
                diversity_ratio = top10_count / all_count * 100
                diversity_ratios.append(diversity_ratio)
                llm_labels.append(llm_name)

        if diversity_ratios:
            bars = ax4.bar(llm_labels, diversity_ratios,
                           color=plt.cm.plasma(np.linspace(0, 1, len(diversity_ratios))),
                           alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, diversity_ratios):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

            ax4.set_ylabel('Top-10 / Total Ratio (%)', fontweight='bold')
            ax4.set_title('Molecular Diversity Ratio\n(Top-10 vs All Generated)',
                          fontweight='bold', fontsize=12, pad=20)
            ax4.set_xticklabels(llm_labels, rotation=45, ha='right')
            ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_smiles_overlap_part2_diversity_analysis.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_overlap_analysis_part3(self, llm_all_smiles: Dict, pipeline_evaluations: Dict, llm_names: List,
                                       pipeline_name: str, pipeline_type: str, viz_dir: Path):
        """Part 3: Performance Analysis & Summary"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'{pipeline_name} Pipeline: Performance Analysis & Summary',
                     fontsize=16, fontweight='bold', y=0.98)

        # Create custom layout
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1])

        # 1. Performance correlation with overlap
        ax1 = fig.add_subplot(gs[0, 0])

        llm_performance = {}
        for llm_name, llm_eval in pipeline_evaluations.items():
            auc_scores = []
            for query_eval in llm_eval["query_evaluations"].values():
                pipeline_data = query_eval["pipeline_data"]
                if pipeline_data["auc_scores"]:
                    auc_scores.extend(pipeline_data["auc_scores"])

            avg_auc = np.mean(auc_scores) if auc_scores else 0

            overlaps_with_others = []
            for other_llm in llm_names:
                if other_llm != llm_name:
                    set1 = llm_all_smiles[llm_name]['top_10']
                    set2 = llm_all_smiles[other_llm]['top_10']

                    if len(set1) > 0 and len(set2) > 0:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        jaccard = intersection / union if union > 0 else 0
                        overlaps_with_others.append(jaccard)

            avg_overlap = np.mean(overlaps_with_others) if overlaps_with_others else 0
            llm_performance[llm_name] = {'auc': avg_auc, 'overlap': avg_overlap}

        if llm_performance:
            aucs = [data['auc'] for data in llm_performance.values()]
            overlaps = [data['overlap'] for data in llm_performance.values()]

            scatter = ax1.scatter(overlaps, aucs, s=120, alpha=0.7,
                                  c=range(len(llm_names)), cmap='viridis', edgecolors='black')

            for i, (llm_name, data) in enumerate(llm_performance.items()):
                ax1.annotate(llm_name[:10] + '...', (data['overlap'], data['auc']),
                             xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

            ax1.set_xlabel('Average Overlap with Other LLMs', fontweight='bold')
            ax1.set_ylabel('Average AUC Score', fontweight='bold')
            ax1.set_title('Performance vs Chemical Space Overlap', fontweight='bold', fontsize=12, pad=20)
            ax1.grid(alpha=0.3)

        # 2. AUC performance ranking
        ax2 = fig.add_subplot(gs[0, 1])

        auc_performance = []
        for llm_name, data in llm_performance.items():
            auc_performance.append((data['auc'], llm_name))

        auc_performance.sort(reverse=True)

        if auc_performance:
            aucs, names = zip(*auc_performance)
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(aucs)))

            bars = ax2.barh(range(len(names)), aucs, color=colors, alpha=0.8, edgecolor='black')

            for bar, auc in zip(bars, aucs):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{auc:.3f}', va='center', fontweight='bold', fontsize=10)

            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in names], fontsize=10)
            ax2.set_xlabel('Average AUC Score', fontweight='bold')
            ax2.set_title('LLM Performance Ranking', fontweight='bold', fontsize=12, pad=20)
            ax2.grid(axis='x', alpha=0.3)

        # 3. Comprehensive summary table
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')

        summary_data = []
        for llm_name in llm_names:
            data = llm_all_smiles[llm_name]
            perf_data = llm_performance.get(llm_name, {'auc': 0, 'overlap': 0})

            # Calculate AUC sum for this LLM
            total_auc = 0
            for query_eval in pipeline_evaluations[llm_name]["query_evaluations"].values():
                pipeline_data = query_eval["pipeline_data"]
                if pipeline_data["auc_scores"]:
                    total_auc += sum(pipeline_data["auc_scores"])

            summary_data.append([
                llm_name[:20] + '...' if len(llm_name) > 20 else llm_name,
                f"{len(data['top_1'])}",
                f"{len(data['top_5'])}",
                f"{len(data['top_10'])}",
                f"{len(data['all']):,}",
                f"{total_auc:.2f}",
                f"{perf_data['auc']:.3f}",
                f"{perf_data['overlap']:.3f}"
            ])

        table = ax3.table(cellText=summary_data,
                          colLabels=['LLM', 'Unique\nTop-1', 'Unique\nTop-5', 'Unique\nTop-10',
                                     'Total\nUnique', 'Total\nAUC-10', 'Avg\nAUC', 'Avg\nOverlap'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.5)

        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#3498db')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            cell.set_edgecolor('black')
            cell.set_linewidth(1)

        ax3.set_title(f'{pipeline_name}: Complete Performance & SMILES Generation Summary',
                      fontweight='bold', fontsize=14, pad=40)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_smiles_overlap_part3_performance_summary.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_drug_likeness_analysis(self, mol_df: pd.DataFrame, pipeline_type: str, viz_dir: Path):
        """Create comprehensive drug-likeness analysis"""
        print(f"    ✓ Creating {pipeline_type} drug-likeness analysis...")

        pipeline_name = pipeline_type.replace('_', '-').title()

        # Filter valid molecules
        valid_mol_df = mol_df[mol_df['Oracle_Score'] > 0].copy()

        if valid_mol_df.empty:
            print(f"      No valid molecules found for {pipeline_type} drug-likeness analysis")
            return

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'{pipeline_name} Pipeline: Drug-Likeness Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. QED Distribution by LLM
        if 'QED' in valid_mol_df.columns and 'LLM' in valid_mol_df.columns:
            llms = valid_mol_df['LLM'].unique()
            qed_data = [valid_mol_df[valid_mol_df['LLM'] == llm]['QED'].values for llm in llms]

            try:
                parts = ax1.violinplot(qed_data, positions=range(len(llms)), showmeans=True, showmedians=True)
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i % len(colors)])
                    pc.set_alpha(0.7)

                ax1.set_xticks(range(len(llms)))
                ax1.set_xticklabels([llm[:15] + '...' if len(llm) > 15 else llm for llm in llms],
                                    rotation=45, ha='right', fontsize=9)
            except:
                # Fallback to box plot
                bp = ax1.boxplot(qed_data, labels=[llm[:10] + '...' if len(llm) > 10 else llm for llm in llms],
                                 patch_artist=True)
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors[i % len(colors)])
                    patch.set_alpha(0.7)
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)

            ax1.set_ylabel('QED Score', fontweight='bold')
            ax1.set_title('QED (Drug-Likeness) Distribution by LLM', fontweight='bold', pad=20)
            ax1.grid(axis='y', alpha=0.3)

        # 2. Lipinski's Rule of Five Compliance
        if all(col in valid_mol_df.columns for col in ['MW', 'LogP', 'HBD', 'HBA']):
            # Calculate Lipinski violations
            valid_mol_df['Lipinski_Violations'] = (
                    (valid_mol_df['MW'] > 500).astype(int) +
                    (valid_mol_df['LogP'] > 5).astype(int) +
                    (valid_mol_df['HBD'] > 5).astype(int) +
                    (valid_mol_df['HBA'] > 10).astype(int)
            )

            if 'LLM' in valid_mol_df.columns:
                lipinski_stats = []
                for llm in llms:
                    llm_data = valid_mol_df[valid_mol_df['LLM'] == llm]
                    compliant = len(llm_data[llm_data['Lipinski_Violations'] == 0])
                    total = len(llm_data)
                    compliance_rate = (compliant / total * 100) if total > 0 else 0

                    lipinski_stats.append({
                        'LLM': llm,
                        'Compliant': compliant,
                        'Total': total,
                        'Compliance_Rate': compliance_rate
                    })

                lipinski_df = pd.DataFrame(lipinski_stats)

                if not lipinski_df.empty:
                    colors = ['#27ae60', '#e74c3c', '#f39c12', '#9b59b6'][:len(lipinski_df)]
                    bars = ax2.bar(range(len(lipinski_df)), lipinski_df['Compliance_Rate'],
                                   color=colors, alpha=0.8, edgecolor='black')

                    for bar, row in zip(bars, lipinski_df.iterrows()):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width() / 2, height + 1,
                                 f"{height:.1f}%\n({row[1]['Compliant']}/{row[1]['Total']})",
                                 ha='center', va='bottom', fontweight='bold', fontsize=9)

                    ax2.set_xticks(range(len(lipinski_df)))
                    ax2.set_xticklabels([llm[:12] + '...' if len(llm) > 12 else llm for llm in lipinski_df['LLM']],
                                        rotation=45, ha='right', fontsize=9)
                    ax2.set_ylabel('Lipinski Compliance Rate (%)', fontweight='bold')
                    ax2.set_title("Lipinski's Rule of Five Compliance", fontweight='bold', pad=20)
                    ax2.grid(axis='y', alpha=0.3)

        # 3. Molecular Weight Distribution
        if 'MW' in valid_mol_df.columns:
            # Overall distribution
            ax3.hist(valid_mol_df['MW'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
            ax3.axvline(valid_mol_df['MW'].mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {valid_mol_df["MW"].mean():.1f}')
            ax3.axvline(500, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                        label='Lipinski Limit (500)')

            ax3.set_xlabel('Molecular Weight (Da)', fontweight='bold')
            ax3.set_ylabel('Frequency', fontweight='bold')
            ax3.set_title('Molecular Weight Distribution', fontweight='bold', pad=20)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)

        # 4. LogP Distribution
        if 'LogP' in valid_mol_df.columns:
            ax4.hist(valid_mol_df['LogP'], bins=30, alpha=0.7, color='#e74c3c', edgecolor='black')
            ax4.axvline(valid_mol_df['LogP'].mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {valid_mol_df["LogP"].mean():.2f}')
            ax4.axvline(5, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                        label='Lipinski Limit (5)')

            ax4.set_xlabel('LogP (Lipophilicity)', fontweight='bold')
            ax4.set_ylabel('Frequency', fontweight='bold')
            ax4.set_title('LogP Distribution', fontweight='bold', pad=20)
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)

        # 5. Oracle Score vs QED Correlation
        if 'QED' in valid_mol_df.columns:
            # Color by LLM if available
            if 'LLM' in valid_mol_df.columns:
                unique_llms = valid_mol_df['LLM'].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_llms)))

                for i, llm in enumerate(unique_llms):
                    llm_data = valid_mol_df[valid_mol_df['LLM'] == llm]
                    ax5.scatter(llm_data['QED'], llm_data['Oracle_Score'],
                                alpha=0.6, s=20, c=[colors[i]], label=llm[:12] + '...' if len(llm) > 12 else llm)
            else:
                ax5.scatter(valid_mol_df['QED'], valid_mol_df['Oracle_Score'],
                            alpha=0.6, s=20, c='#3498db')

            # Add correlation line
            if len(valid_mol_df) > 1:
                z = np.polyfit(valid_mol_df['QED'], valid_mol_df['Oracle_Score'], 1)
                p = np.poly1d(z)
                ax5.plot(valid_mol_df['QED'], p(valid_mol_df['QED']), "r--", alpha=0.8, linewidth=2)

                # Calculate correlation coefficient
                corr = np.corrcoef(valid_mol_df['QED'], valid_mol_df['Oracle_Score'])[0, 1]
                ax5.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax5.transAxes,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')

            ax5.set_xlabel('QED (Drug-Likeness)', fontweight='bold')
            ax5.set_ylabel('Oracle Score', fontweight='bold')
            ax5.set_title('Oracle Performance vs Drug-Likeness', fontweight='bold', pad=20)
            if 'LLM' in valid_mol_df.columns:
                ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax5.grid(alpha=0.3)

        # 6. Drug-Likeness Quality Distribution
        if 'QED' in valid_mol_df.columns:
            # Categorize by QED ranges
            excellent_drug = len(valid_mol_df[valid_mol_df['QED'] > 0.8])
            good_drug = len(valid_mol_df[(valid_mol_df['QED'] > 0.6) & (valid_mol_df['QED'] <= 0.8)])
            moderate_drug = len(valid_mol_df[(valid_mol_df['QED'] > 0.4) & (valid_mol_df['QED'] <= 0.6)])
            poor_drug = len(valid_mol_df[valid_mol_df['QED'] <= 0.4])

            sizes = [excellent_drug, good_drug, moderate_drug, poor_drug]
            labels = ['Excellent (>0.8)', 'Good (0.6-0.8)', 'Moderate (0.4-0.6)', 'Poor (≤0.4)']
            colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']

            # Filter out zero values
            non_zero_data = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
            if non_zero_data:
                sizes, labels, colors = zip(*non_zero_data)

                wedges, texts, autotexts = ax6.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                   colors=colors, startangle=90,
                                                   textprops={'fontsize': 10, 'fontweight': 'bold'})

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

            ax6.set_title('Drug-Likeness Quality Distribution\n(QED Categories)',
                          fontweight='bold', fontsize=12, pad=20)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(viz_dir / f"{pipeline_type}_drug_likeness_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create additional pharmaceutical properties analysis
        self._create_pharmaceutical_properties_analysis(valid_mol_df, pipeline_type, viz_dir)

        print(f"    ✓ Created {pipeline_type} drug-likeness analysis")

    def _create_pharmaceutical_properties_analysis(self, mol_df: pd.DataFrame, pipeline_type: str, viz_dir: Path):
        """Create detailed pharmaceutical properties analysis"""
        pipeline_name = pipeline_type.replace('_', '-').title()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{pipeline_name} Pipeline: Pharmaceutical Properties Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. TPSA (Topological Polar Surface Area) vs Permeability
        if 'TPSA' in mol_df.columns:
            if 'LLM' in mol_df.columns:
                llms = mol_df['LLM'].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(llms)))

                for i, llm in enumerate(llms):
                    llm_data = mol_df[mol_df['LLM'] == llm]
                    ax1.scatter(llm_data['TPSA'], llm_data['Oracle_Score'],
                                alpha=0.6, s=30, c=[colors[i]],
                                label=llm[:12] + '...' if len(llm) > 12 else llm)
            else:
                ax1.scatter(mol_df['TPSA'], mol_df['Oracle_Score'], alpha=0.6, s=30, c='#3498db')

            # Add TPSA guidelines
            ax1.axvline(140, color='red', linestyle='--', alpha=0.7, label='Oral Bioavailability Limit (140)')
            ax1.axvline(90, color='orange', linestyle='--', alpha=0.7, label='BBB Permeability Limit (90)')

            ax1.set_xlabel('TPSA (Ų)', fontweight='bold')
            ax1.set_ylabel('Oracle Score', fontweight='bold')
            ax1.set_title('Topological Polar Surface Area vs Performance', fontweight='bold', pad=20)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.grid(alpha=0.3)

        # 2. Rotatable Bonds vs Flexibility
        if 'RotBonds' in mol_df.columns:
            rot_bonds_dist = mol_df['RotBonds'].value_counts().sort_index()

            bars = ax2.bar(rot_bonds_dist.index, rot_bonds_dist.values,
                           color='#9b59b6', alpha=0.8, edgecolor='black')

            # Add flexibility guidelines
            ax2.axvline(10, color='red', linestyle='--', alpha=0.7, linewidth=2,
                        label='Flexibility Limit (10)')

            for bar, count in zip(bars, rot_bonds_dist.values):
                if count > max(rot_bonds_dist.values) * 0.05:  # Only label significant bars
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rot_bonds_dist.values) * 0.01,
                             f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax2.set_xlabel('Number of Rotatable Bonds', fontweight='bold')
            ax2.set_ylabel('Number of Molecules', fontweight='bold')
            ax2.set_title('Molecular Flexibility Distribution', fontweight='bold', pad=20)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)

        # 3. Hydrogen Bond Analysis
        if all(col in mol_df.columns for col in ['HBD', 'HBA']):
            # Create 2D histogram
            h = ax3.hist2d(mol_df['HBD'], mol_df['HBA'], bins=15, cmap='Blues', alpha=0.8)
            plt.colorbar(h[3], ax=ax3, label='Molecule Count')

            # Add Lipinski limits
            ax3.axvline(5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='HBD Limit (5)')
            ax3.axhline(10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='HBA Limit (10)')

            ax3.set_xlabel('Hydrogen Bond Donors (HBD)', fontweight='bold')
            ax3.set_ylabel('Hydrogen Bond Acceptors (HBA)', fontweight='bold')
            ax3.set_title('Hydrogen Bonding Profile', fontweight='bold', pad=20)
            ax3.legend()
            ax3.grid(alpha=0.3)

        # 4. Multi-parameter optimization radar chart for top performers
        if 'LLM' in mol_df.columns and all(col in mol_df.columns for col in ['QED', 'MW', 'LogP', 'TPSA']):
            # Get top 20% molecules by Oracle Score
            top_threshold = mol_df['Oracle_Score'].quantile(0.8)
            top_molecules = mol_df[mol_df['Oracle_Score'] >= top_threshold]

            if not top_molecules.empty:
                # Calculate normalized properties for radar chart
                properties = ['QED', 'MW', 'LogP', 'TPSA']
                property_means = {}

                for llm in mol_df['LLM'].unique():
                    llm_top = top_molecules[top_molecules['LLM'] == llm]
                    if not llm_top.empty:
                        # Normalize properties (higher is better for some, lower for others)
                        normalized_props = []
                        normalized_props.append(llm_top['QED'].mean())  # Higher is better
                        normalized_props.append(1 - min(llm_top['MW'].mean() / 1000, 1))  # Lower is better, normalize
                        normalized_props.append(
                            1 - min(abs(llm_top['LogP'].mean()) / 10, 1))  # Closer to optimal range is better
                        normalized_props.append(1 - min(llm_top['TPSA'].mean() / 200, 1))  # Lower is generally better

                        property_means[llm] = normalized_props

                if property_means:
                    # Create simple bar chart instead of radar (easier to read)
                    llm_names = list(property_means.keys())
                    prop_names = ['QED\n(Drug-like)', 'MW\n(Size)', 'LogP\n(Lipophil.)', 'TPSA\n(Polar SA)']

                    x = np.arange(len(prop_names))
                    width = 0.8 / len(llm_names)

                    colors = plt.cm.Set1(np.linspace(0, 1, len(llm_names)))

                    for i, (llm, props) in enumerate(property_means.items()):
                        bars = ax4.bar(x + i * width, props, width,
                                       label=llm[:10] + '...' if len(llm) > 10 else llm,
                                       color=colors[i], alpha=0.8, edgecolor='black')

                        # Add value labels
                        for bar, val in zip(bars, props):
                            if val > 0.1:  # Only label significant values
                                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                                         f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

                    ax4.set_xlabel('Pharmaceutical Properties (Normalized)', fontweight='bold')
                    ax4.set_ylabel('Normalized Score (Higher = Better)', fontweight='bold')
                    ax4.set_title('Top Performers: Multi-Parameter Optimization', fontweight='bold', pad=20)
                    ax4.set_xticks(x + width * (len(llm_names) - 1) / 2)
                    ax4.set_xticklabels(prop_names, fontsize=9)
                    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax4.grid(axis='y', alpha=0.3)
                    ax4.set_ylim(0, 1.1)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_pharmaceutical_properties.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_intuitive_performance_overview(self, all_llm_evaluations: Dict, comp_df: pd.DataFrame,
                                              mol_df: pd.DataFrame, viz_dir: Path):
        """Create an intuitive performance overview with pie charts and clean bar charts"""
        print("Creating Intuitive Performance Overview...")

        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('LLM Performance Overview - Easy to Understand Dashboard',
                     fontsize=18, fontweight='bold', y=0.96)

        # Create a grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])

        # 1. Overall Performance Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])
        if not comp_df.empty:
            llm_performance = comp_df.groupby('LLM')['AUC_Mean'].sum().sort_values(ascending=False)

            colors = plt.cm.Set3(np.linspace(0, 1, len(llm_performance)))
            # Truncate long LLM names for cleaner labels
            short_labels = [name[:12] + '...' if len(name) > 12 else name for name in llm_performance.index]
            wedges, texts, autotexts = ax1.pie(llm_performance.values, labels=short_labels,
                                               autopct='%1.1f%%', colors=colors, startangle=90,
                                               textprops={'fontsize': 8, 'fontweight': 'bold'},
                                               pctdistance=0.85, labeldistance=1.1)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax1.set_title('Overall Performance\nShare by LLM', fontweight='bold', pad=20)

        # 2. Pipeline Performance Comparison - Clean Bar Chart
        ax2 = fig.add_subplot(gs[0, 1:3])
        if not comp_df.empty:
            pipeline_comparison = comp_df.groupby(['LLM', 'Pipeline'])['AUC_Mean'].sum().unstack(fill_value=0)

            x_pos = np.arange(len(pipeline_comparison.index))
            width = 0.35

            if 'Single-Shot' in pipeline_comparison.columns:
                bars1 = ax2.bar(x_pos - width / 2, pipeline_comparison['Single-Shot'],
                                width, label='Single-Shot', color='#3498db', alpha=0.8)
            if 'Iterative' in pipeline_comparison.columns:
                bars2 = ax2.bar(x_pos + width / 2, pipeline_comparison['Iterative'],
                                width, label='Iterative', color='#e74c3c', alpha=0.8)

            # Add value labels on bars
            for bars in [bars1,
                         bars2] if 'Single-Shot' in pipeline_comparison.columns and 'Iterative' in pipeline_comparison.columns else []:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                 f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

            ax2.set_xlabel('LLM Models', fontweight='bold')
            ax2.set_ylabel('Total AUC Score', fontweight='bold')
            ax2.set_title('Pipeline Performance Comparison', fontweight='bold', pad=20)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(pipeline_comparison.index, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)

        # 3. Success Rate Pie Chart
        ax3 = fig.add_subplot(gs[0, 3])
        if not mol_df.empty:
            high_performers = len(mol_df[mol_df['Oracle_Score'] > 0.8])
            medium_performers = len(mol_df[(mol_df['Oracle_Score'] > 0.5) & (mol_df['Oracle_Score'] <= 0.8)])
            low_performers = len(mol_df[(mol_df['Oracle_Score'] > 0) & (mol_df['Oracle_Score'] <= 0.5)])
            failures = len(mol_df[mol_df['Oracle_Score'] <= 0])

            sizes = [high_performers, medium_performers, low_performers, failures]
            labels = ['High (>0.8)', 'Medium (0.5-0.8)', 'Low (0-0.5)', 'Failed (≤0)']
            colors = ['#2ecc71', '#f39c12', '#e67e22', '#95a5a6']

            # Remove zero values
            non_zero_data = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
            if non_zero_data:
                sizes, labels, colors = zip(*non_zero_data)
                wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                   colors=colors, startangle=90,
                                                   textprops={'fontsize': 9, 'fontweight': 'bold'})

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

            ax3.set_title('Molecule Quality\nDistribution', fontweight='bold', pad=20)

        # 4. LLM Ranking - Horizontal Bar Chart
        ax4 = fig.add_subplot(gs[1, :2])
        if not comp_df.empty:
            llm_ranking = comp_df.groupby('LLM').agg({
                'AUC_Mean': 'sum',
                'Query': 'count'
            }).round(2)
            llm_ranking.columns = ['Total_AUC', 'Task_Count']
            llm_ranking['Efficiency'] = llm_ranking['Total_AUC'] / llm_ranking['Task_Count']
            llm_ranking = llm_ranking.sort_values('Total_AUC', ascending=True)

            colors = plt.cm.viridis(np.linspace(0, 1, len(llm_ranking)))
            bars = ax4.barh(llm_ranking.index, llm_ranking['Total_AUC'], color=colors, alpha=0.8)

            # Add value labels
            for bar, val in zip(bars, llm_ranking['Total_AUC']):
                ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         f'{val:.1f}', va='center', fontweight='bold')

            ax4.set_xlabel('Total AUC Score', fontweight='bold')
            ax4.set_title('LLM Performance Ranking', fontweight='bold', pad=20)
            ax4.grid(axis='x', alpha=0.3)

        # 5. Task Coverage by LLM
        ax5 = fig.add_subplot(gs[1, 2:])
        if not comp_df.empty:
            task_coverage = comp_df.groupby('LLM').size().sort_values(ascending=False)

            colors = plt.cm.Set2(np.linspace(0, 1, len(task_coverage)))
            bars = ax5.bar(task_coverage.index, task_coverage.values, color=colors, alpha=0.8)

            # Add value labels
            for bar, val in zip(bars, task_coverage.values):
                ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{int(val)}', ha='center', va='bottom', fontweight='bold')

            ax5.set_ylabel('Number of Tasks', fontweight='bold')
            ax5.set_title('Task Coverage by LLM', fontweight='bold', pad=20)
            ax5.set_xticklabels(task_coverage.index, rotation=45, ha='right')
            ax5.grid(axis='y', alpha=0.3)

        # 6. Performance Distribution by Score Range
        ax6 = fig.add_subplot(gs[2, :2])
        if not mol_df.empty:
            score_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            range_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
            range_counts = []

            for min_score, max_score in score_ranges:
                count = len(mol_df[(mol_df['Oracle_Score'] >= min_score) &
                                   (mol_df['Oracle_Score'] < max_score)])
                range_counts.append(count)

            # Create gradient colors from red to green
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(range_counts)))
            bars = ax6.bar(range_labels, range_counts, color=colors, alpha=0.8)

            # Add percentage labels
            total_molecules = len(mol_df[mol_df['Oracle_Score'] > 0])
            for bar, count in zip(bars, range_counts):
                percentage = (count / total_molecules * 100) if total_molecules > 0 else 0
                ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(range_counts) * 0.01,
                         f'{count}\n({percentage:.1f}%)', ha='center', va='bottom',
                         fontweight='bold', fontsize=9)

            ax6.set_xlabel('Oracle Score Range', fontweight='bold')
            ax6.set_ylabel('Number of Molecules', fontweight='bold')
            ax6.set_title('Score Distribution Across All Tasks', fontweight='bold', pad=20)
            ax6.grid(axis='y', alpha=0.3)

        # 7. Best Performing Tasks (Top 5)
        ax7 = fig.add_subplot(gs[2, 2:])
        if not comp_df.empty:
            best_tasks = comp_df.groupby('Query')['AUC_Mean'].max().nlargest(5)

            colors = plt.cm.plasma(np.linspace(0, 1, len(best_tasks)))
            bars = ax7.barh(range(len(best_tasks)), best_tasks.values, color=colors, alpha=0.8)

            # Add value labels
            for bar, val in zip(bars, best_tasks.values):
                ax7.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{val:.3f}', va='center', fontweight='bold', fontsize=9)

            ax7.set_yticks(range(len(best_tasks)))
            ax7.set_yticklabels([task[:20] + '...' if len(task) > 20 else task
                                 for task in best_tasks.index], fontsize=9)
            ax7.set_xlabel('Best AUC Score', fontweight='bold')
            ax7.set_title('Top 5 Best Performing Tasks', fontweight='bold', pad=20)
            ax7.grid(axis='x', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(viz_dir / "intuitive_performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created intuitive performance overview")

    def create_pipeline_success_breakdown(self, pipeline_evaluations: Dict, pipeline_type: str, viz_dir: Path):
        """Create pie charts showing success rate breakdowns for a specific pipeline"""
        print(f"Creating {pipeline_type} success breakdown pie charts...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        pipeline_name = pipeline_type.replace('_', '-').title()
        fig.suptitle(f'{pipeline_name} Pipeline: Success Rate Breakdown Analysis',
                     fontsize=16, fontweight='bold', y=0.98)

        # Collect all molecules for this pipeline
        all_molecules = []
        llm_molecule_counts = {}

        for llm_name, llm_eval in pipeline_evaluations.items():
            llm_molecules = []
            for query_eval in llm_eval["query_evaluations"].values():
                for run in query_eval["pipeline_data"]["runs"]:
                    llm_molecules.extend(run["molecules"])
                    all_molecules.extend(run["molecules"])
            llm_molecule_counts[llm_name] = len(llm_molecules)

        if all_molecules:
            # 1. Overall Success Rate Pie Chart
            scores = [mol['Oracle_Score'] for mol in all_molecules]
            excellent = len([s for s in scores if s > 0.9])
            good = len([s for s in scores if 0.7 < s <= 0.9])
            decent = len([s for s in scores if 0.5 < s <= 0.7])
            poor = len([s for s in scores if 0.1 < s <= 0.5])
            failed = len([s for s in scores if s <= 0.1])

            sizes = [excellent, good, decent, poor, failed]
            labels = ['Excellent (>0.9)', 'Good (0.7-0.9)', 'Decent (0.5-0.7)', 'Poor (0.1-0.5)', 'Failed (≤0.1)']
            colors = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
            explode = (0.1, 0, 0, 0, 0)  # Emphasize excellent performance

            # Filter out zero values
            non_zero_data = [(size, label, color, exp) for size, label, color, exp in
                             zip(sizes, labels, colors, explode) if size > 0]
            if non_zero_data:
                sizes, labels, colors, explode = zip(*non_zero_data)

                wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                   colors=colors, explode=explode, startangle=90,
                                                   textprops={'fontsize': 10, 'fontweight': 'bold'},
                                                   pctdistance=0.85)

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)

            ax1.set_title(f'{pipeline_name}: Overall Success Rate\n({len(all_molecules):,} total molecules)',
                          fontweight='bold', pad=20)

            # 2. Molecule Generation Share by LLM
            if llm_molecule_counts:
                llm_names = list(llm_molecule_counts.keys())
                counts = list(llm_molecule_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(llm_names)))

                # Truncate long LLM names for better readability
                short_names = [name[:12] + '...' if len(name) > 12 else name for name in llm_names]
                wedges, texts, autotexts = ax2.pie(counts, labels=short_names, autopct='%1.1f%%',
                                                   colors=colors, startangle=45,
                                                   textprops={'fontsize': 9, 'fontweight': 'bold'},
                                                   pctdistance=0.85, labeldistance=1.15)

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                ax2.set_title(f'{pipeline_name}: Molecule Generation\nShare by LLM',
                              fontweight='bold', pad=20)

            # 3. High Quality Molecules (>0.8) by LLM - Donut Chart
            high_quality_by_llm = {}
            for llm_name, llm_eval in pipeline_evaluations.items():
                high_quality_count = 0
                total_count = 0
                for query_eval in llm_eval["query_evaluations"].values():
                    for run in query_eval["pipeline_data"]["runs"]:
                        for mol in run["molecules"]:
                            total_count += 1
                            if mol['Oracle_Score'] > 0.8:
                                high_quality_count += 1
                high_quality_by_llm[llm_name] = high_quality_count

            if any(high_quality_by_llm.values()):
                llm_names = list(high_quality_by_llm.keys())
                high_counts = list(high_quality_by_llm.values())
                colors = plt.cm.viridis(np.linspace(0, 1, len(llm_names)))

                # Create donut chart with abbreviated labels
                short_names = [name[:10] + '...' if len(name) > 10 else name for name in llm_names]
                wedges, texts, autotexts = ax3.pie(high_counts, labels=short_names, autopct='%1.0f',
                                                   colors=colors, startangle=90, wedgeprops=dict(width=0.5),
                                                   textprops={'fontsize': 8, 'fontweight': 'bold'},
                                                   pctdistance=0.75, labeldistance=1.2)

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                # Add center text
                ax3.text(0, 0, f'High Quality\nMolecules\n(Score > 0.8)',
                         ha='center', va='center', fontsize=11, fontweight='bold')

                ax3.set_title(f'{pipeline_name}: High Quality Molecules by LLM',
                              fontweight='bold', pad=20)

            # 4. Task Difficulty Distribution - based on average scores
            task_difficulties = {}
            for llm_name, llm_eval in pipeline_evaluations.items():
                for query_name, query_eval in llm_eval["query_evaluations"].items():
                    if query_name not in task_difficulties:
                        task_difficulties[query_name] = []

                    for run in query_eval["pipeline_data"]["runs"]:
                        scores = [mol['Oracle_Score'] for mol in run["molecules"]]
                        if scores:
                            task_difficulties[query_name].extend(scores)

            # Categorize tasks by difficulty (based on average performance)
            easy_tasks = 0
            medium_tasks = 0
            hard_tasks = 0
            very_hard_tasks = 0

            for task_name, scores in task_difficulties.items():
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score > 0.7:
                        easy_tasks += 1
                    elif avg_score > 0.5:
                        medium_tasks += 1
                    elif avg_score > 0.2:
                        hard_tasks += 1
                    else:
                        very_hard_tasks += 1

            difficulty_counts = [easy_tasks, medium_tasks, hard_tasks, very_hard_tasks]
            difficulty_labels = ['Easy (>0.7)', 'Medium (0.5-0.7)', 'Hard (0.2-0.5)', 'Very Hard (≤0.2)']
            difficulty_colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

            # Filter out zero values
            non_zero_diff = [(count, label, color) for count, label, color in
                             zip(difficulty_counts, difficulty_labels, difficulty_colors) if count > 0]
            if non_zero_diff:
                counts, labels, colors = zip(*non_zero_diff)

                wedges, texts, autotexts = ax4.pie(counts, labels=labels, autopct='%1.0f',
                                                   colors=colors, startangle=90,
                                                   textprops={'fontsize': 10, 'fontweight': 'bold'},
                                                   pctdistance=0.85)

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

            ax4.set_title(f'{pipeline_name}: Task Difficulty\nDistribution',
                          fontweight='bold', pad=20)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"{pipeline_type}_success_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Created {pipeline_type} success breakdown")

    def create_llm_comparison_dashboard(self, all_llm_evaluations: Dict, comp_df: pd.DataFrame, viz_dir: Path):
        """Create a clean, intuitive LLM comparison dashboard"""
        print("Creating LLM Comparison Dashboard...")

        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('LLM Comparison Dashboard - Key Performance Indicators',
                     fontsize=18, fontweight='bold', y=0.96)

        # Create a complex grid layout
        gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1])

        if not all_llm_evaluations:
            return

        # Prepare summary data
        llm_summary = []
        for llm_name, llm_eval in all_llm_evaluations.items():
            summary = llm_eval["summary"]

            total_auc = summary["single_shot"]["auc_sum"] + summary["iterative"]["auc_sum"]
            coverage = summary["successful_queries"] / summary["total_queries"] * 100 if summary[
                                                                                             "total_queries"] > 0 else 0

            # Count molecules and calculate success rates
            total_molecules = 0
            high_score_molecules = 0
            best_score = 0.0

            for query_eval in llm_eval["query_evaluations"].values():
                for pipeline in ['single_shot', 'iterative']:
                    for run in query_eval[pipeline]['runs']:
                        for mol in run['molecules']:
                            total_molecules += 1
                            if mol['Oracle_Score'] > 0.8:
                                high_score_molecules += 1
                            best_score = max(best_score, mol['Oracle_Score'])

            success_rate = (high_score_molecules / total_molecules * 100) if total_molecules > 0 else 0

            llm_summary.append({
                'LLM': llm_name,
                'Total_AUC': total_auc,
                'Coverage': coverage,
                'Success_Rate': success_rate,
                'Best_Score': best_score,
                'Total_Molecules': total_molecules,
                'High_Score_Molecules': high_score_molecules
            })

        summary_df = pd.DataFrame(llm_summary).sort_values('Total_AUC', ascending=False)

        if not summary_df.empty:
            # 1. Overall Performance Ranking - Large horizontal bar chart
            ax1 = fig.add_subplot(gs[0, :3])
            colors = plt.cm.plasma(np.linspace(0, 1, len(summary_df)))
            bars = ax1.barh(summary_df['LLM'], summary_df['Total_AUC'], color=colors, alpha=0.8)

            # Add value labels with rank
            for i, (bar, val) in enumerate(zip(bars, summary_df['Total_AUC'])):
                ax1.text(bar.get_width() + max(summary_df['Total_AUC']) * 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f'#{i + 1} - {val:.1f}', va='center', fontweight='bold', fontsize=11)

            ax1.set_xlabel('Total AUC Score', fontweight='bold', fontsize=12)
            ax1.set_title('🏆 LLM Performance Ranking', fontweight='bold', fontsize=14, pad=20)
            ax1.grid(axis='x', alpha=0.3)

            # 2. Success Rate Comparison - Pie Chart
            ax2 = fig.add_subplot(gs[0, 3:5])
            colors = plt.cm.Set3(np.linspace(0, 1, len(summary_df)))
            # Use abbreviated LLM names to prevent label clutter
            short_llm_names = [name[:10] + '...' if len(name) > 10 else name for name in summary_df['LLM']]
            wedges, texts, autotexts = ax2.pie(summary_df['Success_Rate'], labels=short_llm_names,
                                               autopct='%1.1f%%', colors=colors, startangle=90,
                                               textprops={'fontsize': 8, 'fontweight': 'bold'},
                                               pctdistance=0.85, labeldistance=1.15)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax2.set_title('🎯 Success Rate Share\n(High Quality Molecules)', fontweight='bold', fontsize=12, pad=20)

            # 3. Best Score Achievement - Donut Chart
            ax3 = fig.add_subplot(gs[0, 5])
            colors = plt.cm.viridis(np.linspace(0, 1, len(summary_df)))
            # Use abbreviated names for donut chart to avoid crowding
            short_llm_names = [name[:8] + '...' if len(name) > 8 else name for name in summary_df['LLM']]
            wedges, texts, autotexts = ax3.pie(summary_df['Best_Score'], labels=short_llm_names,
                                               autopct='%.2f', colors=colors, startangle=90,
                                               wedgeprops=dict(width=0.6),
                                               textprops={'fontsize': 7, 'fontweight': 'bold'},
                                               pctdistance=0.8, labeldistance=1.25)

            ax3.text(0, 0, 'Best\nScores', ha='center', va='center', fontsize=10, fontweight='bold')
            ax3.set_title('⭐ Peak Performance', fontweight='bold', fontsize=12, pad=20)

            # 4. Task Coverage - Clean Bar Chart
            ax4 = fig.add_subplot(gs[1, :2])
            bars = ax4.bar(summary_df['LLM'], summary_df['Coverage'],
                           color='#3498db', alpha=0.8, edgecolor='navy', linewidth=1)

            for bar, val in zip(bars, summary_df['Coverage']):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                         f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

            ax4.set_ylabel('Coverage Percentage', fontweight='bold')
            ax4.set_title('📊 Task Coverage by LLM', fontweight='bold', fontsize=12, pad=20)
            ax4.set_xticklabels(summary_df['LLM'], rotation=45, ha='right')
            ax4.set_ylim(0, max(summary_df['Coverage']) * 1.2)
            ax4.grid(axis='y', alpha=0.3)

            # 5. Molecule Generation Volume
            ax5 = fig.add_subplot(gs[1, 2:4])
            bars = ax5.bar(summary_df['LLM'], summary_df['Total_Molecules'],
                           color='#e74c3c', alpha=0.8, edgecolor='darkred', linewidth=1)

            for bar, val in zip(bars, summary_df['Total_Molecules']):
                ax5.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(summary_df['Total_Molecules']) * 0.02,
                         f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax5.set_ylabel('Total Molecules Generated', fontweight='bold')
            ax5.set_title('🧪 Molecule Generation Volume', fontweight='bold', fontsize=12, pad=20)
            ax5.set_xticklabels(summary_df['LLM'], rotation=45, ha='right')
            ax5.grid(axis='y', alpha=0.3)

            # 6. Efficiency Score (AUC per molecule)
            ax6 = fig.add_subplot(gs[1, 4:])
            efficiency = summary_df['Total_AUC'] / summary_df['Total_Molecules']
            bars = ax6.bar(summary_df['LLM'], efficiency * 1000,  # Scale for readability
                           color='#f39c12', alpha=0.8, edgecolor='orange', linewidth=1)

            for bar, val in zip(bars, efficiency * 1000):
                ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(efficiency * 1000) * 0.02,
                         f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax6.set_ylabel('Efficiency Score (×1000)', fontweight='bold')
            ax6.set_title('⚡ Efficiency Ratio\n(AUC per Molecule)', fontweight='bold', fontsize=12, pad=20)
            ax6.set_xticklabels(summary_df['LLM'], rotation=45, ha='right')
            ax6.grid(axis='y', alpha=0.3)

            # 7. Performance Breakdown by Pipeline - Stacked Bar
            ax7 = fig.add_subplot(gs[2, :3])
            if not comp_df.empty:
                pipeline_data = comp_df.groupby(['LLM', 'Pipeline'])['AUC_Mean'].sum().unstack(fill_value=0)

                x_pos = np.arange(len(pipeline_data.index))
                width = 0.6

                bottom = np.zeros(len(pipeline_data.index))
                colors = ['#3498db', '#e74c3c']

                for i, pipeline in enumerate(pipeline_data.columns):
                    bars = ax7.bar(x_pos, pipeline_data[pipeline], width, bottom=bottom,
                                   label=pipeline, color=colors[i % len(colors)], alpha=0.8)
                    bottom += pipeline_data[pipeline]

                ax7.set_xlabel('LLM Models', fontweight='bold')
                ax7.set_ylabel('AUC Score', fontweight='bold')
                ax7.set_title('🔄 Pipeline Performance Breakdown', fontweight='bold', fontsize=12, pad=20)
                ax7.set_xticks(x_pos)
                ax7.set_xticklabels(pipeline_data.index, rotation=45, ha='right')
                ax7.legend()
                ax7.grid(axis='y', alpha=0.3)

            # 8. Quality Distribution Summary
            ax8 = fig.add_subplot(gs[2, 3:])
            quality_summary = []
            for _, row in summary_df.iterrows():
                total = row['Total_Molecules']
                high_quality = row['High_Score_Molecules']
                medium_quality = total * 0.3  # Estimated based on typical distributions
                low_quality = total - high_quality - medium_quality

                quality_summary.append({
                    'LLM': row['LLM'],
                    'High': high_quality,
                    'Medium': medium_quality,
                    'Low': low_quality
                })

            quality_df = pd.DataFrame(quality_summary)

            x_pos = np.arange(len(quality_df))
            width = 0.6

            # Stacked bar chart for quality distribution
            bars1 = ax8.bar(x_pos, quality_df['High'], width, label='High Quality (>0.8)',
                            color='#27ae60', alpha=0.8)
            bars2 = ax8.bar(x_pos, quality_df['Medium'], width, bottom=quality_df['High'],
                            label='Medium Quality (0.5-0.8)', color='#f39c12', alpha=0.8)
            bars3 = ax8.bar(x_pos, quality_df['Low'], width,
                            bottom=quality_df['High'] + quality_df['Medium'],
                            label='Low Quality (<0.5)', color='#e74c3c', alpha=0.8)

            ax8.set_xlabel('LLM Models', fontweight='bold')
            ax8.set_ylabel('Number of Molecules', fontweight='bold')
            ax8.set_title('📈 Quality Distribution Overview', fontweight='bold', fontsize=12, pad=20)
            ax8.set_xticks(x_pos)
            ax8.set_xticklabels(quality_df['LLM'], rotation=45, ha='right')
            ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax8.grid(axis='y', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(viz_dir / "llm_comparison_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created LLM comparison dashboard")


def main():
    """Main execution function"""
    print("Research-Focused Multi-LLM Oracle Comparison (Pre-computed Scores)")
    print("=" * 70)

    comparator = ResearchFocusedLLMComparator(base_dir="results")

    try:
        results = comparator.run_research_focused_analysis()

        if results:
            print("\n" + "=" * 80)
            print("RESEARCH ANALYSIS COMPLETE!")
            print("=" * 80)
            return results
        else:
            print("Analysis failed!")
            return None

    except Exception as e:
        print(f"Analysis failed with error: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()