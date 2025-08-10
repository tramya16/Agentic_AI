import json
import numpy as np
from pathlib import Path
from molecular_metrics import MolecularMetrics
from collections import defaultdict

class SMILESOverlapAnalyzer:
    def __init__(self):
        self.metrics = MolecularMetrics()
    
    def analyze_results_directory(self, results_dir, top_n=10):
        """Analyze overlap in a results directory"""
        results_dir = Path(results_dir)
        
        # Find all detailed result files
        result_files = list(results_dir.glob("**/*_detailed_*.json"))
        
        overlap_analysis = {}
        
        for result_file in result_files:
            print(f"📄 Analyzing: {result_file.name}")
            
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            query_name = data.get("query_name", "unknown")
            
            # Extract molecules from single-shot and iterative
            ss_molecules = []
            it_molecules = []
            
            for run_data in data.get("single_shot", []):
                if not run_data.get("error"):
                    molecules_data = run_data.get("molecules_data", {})
                    ss_molecules.extend(molecules_data.get("valid_molecules", []))
            
            for run_data in data.get("iterative", []):
                if not run_data.get("error"):
                    molecules_data = run_data.get("molecules_data", {})
                    it_molecules.extend(molecules_data.get("valid_molecules", []))
            
            if ss_molecules and it_molecules:
                # Calculate overlap
                top_n_overlap = self.metrics.calculate_top_n_overlap(
                    ss_molecules, it_molecules, n=top_n
                )
                
                positional_overlap = self.metrics.calculate_positional_overlap(
                    ss_molecules, it_molecules, n=top_n
                )
                
                overlap_analysis[query_name] = {
                    "single_shot_count": len(ss_molecules),
                    "iterative_count": len(it_molecules),
                    "top_n_overlap": top_n_overlap,
                    "positional_overlap": positional_overlap
                }
        
        return overlap_analysis
    
    def compare_models(self, results_dirs, model_names):
        """Compare SMILES overlap between different models"""
        model_molecules = {}
        
        for i, results_dir in enumerate(results_dirs):
            model_name = model_names[i]
            results_dir = Path(results_dir)
            
            model_molecules[model_name] = defaultdict(list)
            
            # Extract molecules by query
            result_files = list(results_dir.glob("**/*_detailed_*.json"))
            
            for result_file in result_files:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                query_name = data.get("query_name", "unknown")
                
                # Get all valid molecules
                all_molecules = []
                for pipeline in ["single_shot", "iterative"]:
                    for run_data in data.get(pipeline, []):
                        if not run_data.get("error"):
                            molecules_data = run_data.get("molecules_data", {})
                            all_molecules.extend(molecules_data.get("valid_molecules", []))
                
                model_molecules[model_name][query_name] = list(set(all_molecules))
        
        # Calculate cross-model overlaps
        cross_model_analysis = {}
        model_list = list(model_molecules.keys())
        
        for i, model1 in enumerate(model_list):
            for model2 in model_list[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                cross_model_analysis[comparison_key] = {}
                
                # Compare by query
                for query_name in model_molecules[model1].keys():
                    if query_name in model_molecules[model2]:
                        mol1 = model_molecules[model1][query_name]
                        mol2 = model_molecules[model2][query_name]
                        
                        if mol1 and mol2:
                            overlap = self.metrics.calculate_top_n_overlap(mol1, mol2, n=10)
                            cross_model_analysis[comparison_key][query_name] = overlap
        
        return cross_model_analysis
    
    def print_overlap_summary(self, overlap_analysis):
        """Print a summary of overlap analysis"""
        print("\n" + "="*80)
        print("🔍 SMILES OVERLAP ANALYSIS SUMMARY")
        print("="*80)
        
        for query_name, analysis in overlap_analysis.items():
            print(f"\n📊 Query: {query_name}")
            print(f"   Single-shot molecules: {analysis['single_shot_count']}")
            print(f"   Iterative molecules: {analysis['iterative_count']}")
            
            top_n = analysis['top_n_overlap']
            print(f"   Top-{top_n['top_n']} overlap: {top_n['overlap_count']}/{top_n['max_possible_overlap']} ({top_n['overlap_percentage']*100:.1f}%)")
            print(f"   Jaccard similarity: {top_n['jaccard_similarity']:.3f}")
            
            pos_overlap = analysis['positional_overlap']
            print(f"   Positional matches: {pos_overlap['positional_matches']}/{pos_overlap['total_positions_compared']}")
            
            if top_n['overlap_molecules']:
                print(f"   Overlapping molecules: {top_n['overlap_molecules'][:3]}")

# Usage function
def run_overlap_analysis(results_dir, top_n=10):
    analyzer = SMILESOverlapAnalyzer()
    
    print(f"🔍 Running SMILES overlap analysis on: {results_dir}")
    
    # Analyze single directory
    overlap_results = analyzer.analyze_results_directory(results_dir, top_n=top_n)
    analyzer.print_overlap_summary(overlap_results)
    
    # Save results
    output_file = Path(results_dir) / "smiles_overlap_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(overlap_results, f, indent=2, default=str)
    
    print(f"\n💾 Overlap analysis saved to: {output_file}")
    return overlap_results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        run_overlap_analysis(results_dir, top_n)
    else:
        print("Usage: python scripts/overlap_analyzer.py <results_directory> [top_n]")
