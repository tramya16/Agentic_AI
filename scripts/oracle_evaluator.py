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
import warnings
from rdkit import RDLogger

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# Oracle mapping
ORACLE_MAPPING = {
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


class OracleEvaluator:
    def __init__(self):
        self.oracles = {}
        self.load_oracles()
    
    def load_oracles(self):
        """Load TDC oracles"""
        print("🔮 Loading TDC oracles...")
        
        for query_name, oracle_name in ORACLE_MAPPING.items():
            try:
                oracle = Oracle(name=oracle_name)
                self.oracles[query_name] = oracle
                print(f"✅ Loaded {oracle_name}")
            except Exception as e:
                print(f"❌ Failed to load {oracle_name}: {e}")
                self.oracles[query_name] = None
        
        successful = len([o for o in self.oracles.values() if o is not None])
        print(f"📊 Loaded {successful}/{len(ORACLE_MAPPING)} oracles")
    
    def evaluate_results_directory(self, results_dir):
        """Evaluate all results in a directory"""
        results_dir = Path(results_dir)
        print(f"📁 Evaluating results in: {results_dir}")
        
        # Find all experiment summary files
        summary_files = list(results_dir.glob("**/experiment_summary.json"))
        
        if not summary_files:
            print("❌ No experiment summary files found")
            return None
        
        all_evaluations = {}
        
        for summary_file in summary_files:
            print(f"📄 Processing: {summary_file}")
            
            try:
                with open(summary_file, 'r') as f:
                    experiment_data = json.load(f)
                
                # Extract experiment info
                model_name = experiment_data['metadata']['model_name']
                temperature = experiment_data['temperature']
                
                # Evaluate each query result
                query_evaluations = {}
                
                for result in experiment_data['results']:
                    query_name = result['query_name']
                    evaluation = self._evaluate_query_result(query_name, result)
                    
                    if evaluation:
                        query_evaluations[query_name] = evaluation
                
                # Store evaluations
                config_key = f"{model_name}_temp_{str(temperature).replace('.', '_')}"
                all_evaluations[config_key] = {
                    "model_name": model_name,
                    "temperature": temperature,
                    "results_dir": str(summary_file.parent),
                    "query_evaluations": query_evaluations,
                    "summary": self._create_summary_stats(query_evaluations)
                }
                
                # Save individual evaluation
                eval_file = summary_file.parent / "evaluations" / "oracle_evaluation.json"
                eval_file.parent.mkdir(exist_ok=True)
                
                with open(eval_file, 'w') as f:
                    json.dump(all_evaluations[config_key], f, indent=2, default=str)
                
                print(f"✅ Evaluated {len(query_evaluations)} queries for {config_key}")
                
            except Exception as e:
                print(f"❌ Error processing {summary_file}: {e}")
                continue
        
        return all_evaluations
    
    def _evaluate_query_result(self, query_name, result_data):
        """Evaluate a single query result"""
        if query_name not in self.oracles or self.oracles[query_name] is None:
            return None
        
        oracle = self.oracles[query_name]
        oracle_name = ORACLE_MAPPING[query_name]
        
        evaluation = {
            "query_name": query_name,
            "oracle_name": oracle_name,
            "single_shot": {"runs": [], "auc_scores": [], "best_scores": []},
            "iterative": {"runs": [], "auc_scores": [], "best_scores": []}
        }
        
        # Evaluate single-shot runs
        for run_data in result_data.get("single_shot", []):
            if run_data.get("error"):
                continue
                
            molecules = run_data.get("molecules_data", {}).get("valid_molecules", [])
            if not molecules:
                continue
            
            scores = []
            for smiles in molecules:
                try:
                    score = oracle(smiles)
                    scores.append(float(score) if score is not None else 0.0)
                except:
                    scores.append(0.0)
            
            if scores:
                auc_score = self._calculate_auc_top_k(scores, k=10)
                best_score = max(scores)
                
                evaluation["single_shot"]["runs"].append({
                    "run": run_data.get("run", 1),
                    "scores": scores,
                    "auc_top_10": auc_score,
                    "best_score": best_score,
                    "total_molecules": len(scores)
                })
                
                evaluation["single_shot"]["auc_scores"].append(auc_score)
                evaluation["single_shot"]["best_scores"].append(best_score)
        
        # Evaluate iterative runs
        for run_data in result_data.get("iterative", []):
            if run_data.get("error"):
                continue
                
            molecules = run_data.get("molecules_data", {}).get("valid_molecules", [])
            if not molecules:
                continue
            
            scores = []
            for smiles in molecules:
                try:
                    score = oracle(smiles)
                    scores.append(float(score) if score is not None else 0.0)
                except:
                    scores.append(0.0)
            
            if scores:
                auc_score = self._calculate_auc_top_k(scores, k=10)
                best_score = max(scores)
                
                evaluation["iterative"]["runs"].append({
                    "run": run_data.get("run", 1),
                    "scores": scores,
                    "auc_top_10": auc_score,
                    "best_score": best_score,
                    "total_molecules": len(scores)
                })
                
                evaluation["iterative"]["auc_scores"].append(auc_score)
                evaluation["iterative"]["best_scores"].append(best_score)
        
        return evaluation
    
    def _calculate_auc_top_k(self, scores, k=10):
        """Calculate AUC for top-k molecules"""
        if len(scores) == 0:
            return 0.0
        
        sorted_scores = sorted(scores, reverse=True)
        top_k_scores = sorted_scores[:min(k, len(sorted_scores))]
        
        if len(top_k_scores) < 2:
            return np.mean(top_k_scores) if top_k_scores else 0.0
        
        x = np.linspace(0, 1, len(top_k_scores))
        
        try:
            from sklearn.metrics import auc
            return auc(x, top_k_scores)
        except:
            return np.mean(top_k_scores)
    
    def _create_summary_stats(self, query_evaluations):
        """Create summary statistics"""
        if not query_evaluations:
            return {}
        
        ss_aucs = []
        it_aucs = []
        ss_bests = []
        it_bests = []
        
        for eval_data in query_evaluations.values():
            ss_aucs.extend(eval_data["single_shot"]["auc_scores"])
            it_aucs.extend(eval_data["iterative"]["auc_scores"])
            ss_bests.extend(eval_data["single_shot"]["best_scores"])
            it_bests.extend(eval_data["iterative"]["best_scores"])
        
        return {
            "single_shot": {
                "auc_mean": np.mean(ss_aucs) if ss_aucs else 0.0,
                "auc_sum": np.sum(ss_aucs) if ss_aucs else 0.0,
                "best_max": np.max(ss_bests) if ss_bests else 0.0,
                "queries_evaluated": len([e for e in query_evaluations.values() if e["single_shot"]["auc_scores"]])
            },
            "iterative": {
                "auc_mean": np.mean(it_aucs) if it_aucs else 0.0,
                "auc_sum": np.sum(it_aucs) if it_aucs else 0.0,
                "best_max": np.max(it_bests) if it_bests else 0.0,
                "queries_evaluated": len([e for e in query_evaluations.values() if e["iterative"]["auc_scores"]])
            }
        }
    
    def create_comparison_visualizations(self, all_evaluations, output_dir):
        """Create comparison visualizations across all models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"📊 Creating visualizations in: {output_dir}")
        
        # Prepare comparison data
        comparison_data = []
        
        for config_name, eval_data in all_evaluations.items():
            model_name = eval_data["model_name"]
            temperature = eval_data["temperature"]
            summary = eval_data["summary"]
            
            comparison_data.append({
                "Config": config_name,
                "Model": model_name,
                "Temperature": temperature,
                "SS_AUC_Sum": summary["single_shot"]["auc_sum"],
                "IT_AUC_Sum": summary["iterative"]["auc_sum"],
                "SS_AUC_Mean": summary["single_shot"]["auc_mean"],
                "IT_AUC_Mean": summary["iterative"]["auc_mean"],
                "SS_Best": summary["single_shot"]["best_max"],
                "IT_Best": summary["iterative"]["best_max"],
                "SS_Queries": summary["single_shot"]["queries_evaluated"],
                "IT_Queries": summary["iterative"]["queries_evaluated"]
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold')
        
        # AUC Sum comparison
        ax1 = axes[0, 0]
        x = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x - width/2, df['SS_AUC_Sum'], width, label='Single-Shot', alpha=0.8)
        ax1.bar(x + width/2, df['IT_AUC_Sum'], width, label='Iterative', alpha=0.8)
        ax1.set_xlabel('Model Configuration')
        ax1.set_ylabel('AUC Sum')
        ax1.set_title('Total AUC Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Config'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Best score comparison
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, df['SS_Best'], width, label='Single-Shot', alpha=0.8)
        ax2.bar(x + width/2, df['IT_Best'], width, label='Iterative', alpha=0.8)
        ax2.set_xlabel('Model Configuration')
        ax2.set_ylabel('Best Score')
        ax2.set_title('Best Individual Scores')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['Config'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Temperature effect
        ax3 = axes[0, 2]
        temp_data = df.groupby('Temperature').agg({
            'SS_AUC_Sum': 'mean',
            'IT_AUC_Sum': 'mean'
        }).reset_index()
        
        ax3.plot(temp_data['Temperature'], temp_data['SS_AUC_Sum'], 'o-', label='Single-Shot', linewidth=2, markersize=8)
        ax3.plot(temp_data['Temperature'], temp_data['IT_AUC_Sum'], 's-', label='Iterative', linewidth=2, markersize=8)
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Average AUC Sum')
        ax3.set_title('Temperature Effect on Performance')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Model comparison
        ax4 = axes[1, 0]
        model_data = df.groupby('Model').agg({
            'SS_AUC_Sum': 'mean',
            'IT_AUC_Sum': 'mean'
        }).reset_index()
        
        x_models = np.arange(len(model_data))
        ax4.bar(x_models - width/2, model_data['SS_AUC_Sum'], width, label='Single-Shot', alpha=0.8)
        ax4.bar(x_models + width/2, model_data['IT_AUC_Sum'], width, label='Iterative', alpha=0.8)
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Average AUC Sum')
        ax4.set_title('Model Performance Comparison')
        ax4.set_xticks(x_models)
        ax4.set_xticklabels(model_data['Model'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # Pipeline comparison
        ax5 = axes[1, 1]
        pipeline_comparison = pd.DataFrame({
            'Pipeline': ['Single-Shot', 'Iterative'],
            'AUC_Sum': [df['SS_AUC_Sum'].mean(), df['IT_AUC_Sum'].mean()],
            'Best_Score': [df['SS_Best'].mean(), df['IT_Best'].mean()]
        })
        
        x_pipe = np.arange(len(pipeline_comparison))
        ax5_twin = ax5.twinx()
        
        bars1 = ax5.bar(x_pipe - width/2, pipeline_comparison['AUC_Sum'], width, 
                       label='AUC Sum', alpha=0.8, color='skyblue')
        bars2 = ax5_twin.bar(x_pipe + width/2, pipeline_comparison['Best_Score'], width, 
                            label='Best Score', alpha=0.8, color='lightcoral')
        
        ax5.set_xlabel('Pipeline')
        ax5.set_ylabel('Average AUC Sum', color='skyblue')
        ax5_twin.set_ylabel('Average Best Score', color='lightcoral')
        ax5.set_title('Overall Pipeline Comparison')
        ax5.set_xticks(x_pipe)
        ax5.set_xticklabels(pipeline_comparison['Pipeline'])
        
        # Success rate
        ax6 = axes[1, 2]
        df['SS_Success_Rate'] = df['SS_Queries'] / len(ORACLE_MAPPING) * 100
        df['IT_Success_Rate'] = df['IT_Queries'] / len(ORACLE_MAPPING) * 100
        
        ax6.bar(x - width/2, df['SS_Success_Rate'], width, label='Single-Shot', alpha=0.8)
        ax6.bar(x + width/2, df['IT_Success_Rate'], width, label='Iterative', alpha=0.8)
        ax6.set_xlabel('Model Configuration')
        ax6.set_ylabel('Success Rate (%)')
        ax6.set_title('Query Success Rate')
        ax6.set_xticks(x)
        ax6.set_xticklabels(df['Config'], rotation=45, ha='right')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison table
        comparison_file = output_dir / "model_comparison.csv"
        df.to_csv(comparison_file, index=False)
        
        print(f"📊 Visualizations saved to: {output_dir}")
        print(f"📋 Comparison data saved to: {comparison_file}")
        
        return df
