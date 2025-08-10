# molecular_metrics.py (ENHANCED WITH NOVELTY)

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd

# Set matplotlib style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style if seaborn not available


class MolecularMetrics:
    """
    Implements simplified metrics from 'Molecular Sets (MOSES): A Benchmarking Platform
    for Molecular Generation Models' for evaluating molecular generation quality.
    Enhanced with ZINC250k novelty checking.
    """

    def __init__(self):
        self.valid_molecules = []
        self.invalid_molecules = []
        self._zinc_cache = None  # Cache ZINC250k dataset

    def calculate_validity(self, smiles_list: List[str]) -> Dict:
        """Calculate validity metrics"""
        valid_count = 0
        valid_mols = []
        invalid_smiles = []

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
                    valid_mols.append(mol)
                else:
                    invalid_smiles.append(smiles)
            except:
                invalid_smiles.append(smiles)

        total = len(smiles_list)
        validity_ratio = valid_count / total if total > 0 else 0

        self.valid_molecules = valid_mols
        self.invalid_molecules = invalid_smiles

        return {
            "validity": validity_ratio,
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "total_count": total
        }

    def calculate_uniqueness(self, smiles_list: List[str]) -> Dict:
        """Calculate uniqueness metrics"""
        # Convert to canonical SMILES for proper comparison
        canonical_smiles = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    canonical_smiles.append(canonical)
            except:
                continue

        total_valid = len(canonical_smiles)
        unique_smiles = list(set(canonical_smiles))
        unique_count = len(unique_smiles)

        uniqueness_ratio = unique_count / total_valid if total_valid > 0 else 0

        return {
            "uniqueness": uniqueness_ratio,
            "unique_count": unique_count,
            "total_valid": total_valid,
            "duplicate_count": total_valid - unique_count
        }

    def calculate_novelty(self, generated_smiles: List[str], reference_smiles: List[str]) -> Dict:
        """Calculate novelty (how many generated molecules are not in reference set)"""
        gen_canonical = set()
        for smiles in generated_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    gen_canonical.add(Chem.MolToSmiles(mol))
            except:
                continue

        ref_canonical = set()
        for smiles in reference_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    ref_canonical.add(Chem.MolToSmiles(mol))
            except:
                continue

        novel_molecules = gen_canonical - ref_canonical
        novelty_ratio = len(novel_molecules) / len(gen_canonical) if gen_canonical else 0

        return {
            "novelty": novelty_ratio,
            "novel_count": len(novel_molecules),
            "total_generated": len(gen_canonical),
            "overlap_count": len(gen_canonical & ref_canonical)
        }

    def calculate_enhanced_novelty(self, generated_smiles: List[str], reference_dataset: str = "zinc250k") -> Dict:
        """
        Enhanced novelty calculation using standard reference datasets.

        Args:
            generated_smiles: List of generated SMILES
            reference_dataset: Reference dataset to use ("zinc250k", "chembl_drugs", "moses")

        Returns:
            Dictionary with enhanced novelty metrics including dataset statistics
        """
        try:
            from utils.chemistry_utils import calculate_novelty_against_zinc, download_zinc250k_smiles

            if reference_dataset == "zinc250k":
                # Load ZINC250k dataset (cached)
                if self._zinc_cache is None:
                    print("Loading ZINC250k dataset for novelty calculation...")
                    self._zinc_cache = download_zinc250k_smiles()

                if not self._zinc_cache:
                    return {
                        "status": "error",
                        "error": "Failed to load ZINC250k dataset",
                        "enhanced_novelty": 0.0,
                        "reference_dataset": reference_dataset
                    }

                # Calculate novelty against ZINC250k
                novelty_result = calculate_novelty_against_zinc(generated_smiles, self._zinc_cache)

                if novelty_result["status"] == "success":
                    return {
                        "status": "success",
                        "enhanced_novelty": novelty_result["novelty_rate"],
                        "novel_count": novelty_result["novel_count"],
                        "known_count": novelty_result["known_count"],
                        "reference_dataset": reference_dataset,
                        "reference_size": novelty_result["zinc_reference_size"],
                        "novel_molecules": novelty_result["novel_molecules"][:10],  # Top 10 for display
                        "known_molecules": novelty_result["known_molecules"][:10],  # Top 10 for display
                        "total_generated": novelty_result["total_generated"],
                        "valid_generated": novelty_result["valid_generated"],
                        "invalid_generated": novelty_result["invalid_generated"]
                    }
                else:
                    return {
                        "status": "error",
                        "error": novelty_result["error"],
                        "enhanced_novelty": 0.0,
                        "reference_dataset": reference_dataset
                    }

            else:
                return {
                    "status": "error",
                    "error": f"Reference dataset '{reference_dataset}' not yet implemented",
                    "enhanced_novelty": 0.0,
                    "reference_dataset": reference_dataset
                }

        except ImportError as e:
            return {
                "status": "error",
                "error": f"Required chemistry utils not available: {str(e)}",
                "enhanced_novelty": 0.0,
                "reference_dataset": reference_dataset
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Enhanced novelty calculation failed: {str(e)}",
                "enhanced_novelty": 0.0,
                "reference_dataset": reference_dataset
            }

    def calculate_diversity(self, smiles_list: List[str]) -> Dict:
        """Calculate internal diversity using Tanimoto distances"""
        try:
            from rdkit import DataStructs
            from rdkit.Chem import rdMolDescriptors
        except ImportError:
            return {"diversity": 0.0, "pairwise_distances": [], "num_comparisons": 0}

        mols = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mols.append(mol)
            except:
                continue

        if len(mols) < 2:
            return {"diversity": 0.0, "pairwise_distances": [], "num_comparisons": 0}

        try:
            # Calculate Morgan fingerprints
            fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]

            # Calculate pairwise Tanimoto distances
            distances = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    distance = 1 - similarity
                    distances.append(distance)

            avg_diversity = np.mean(distances) if distances else 0.0

            return {
                "diversity": avg_diversity,
                "pairwise_distances": distances,
                "num_comparisons": len(distances)
            }
        except:
            return {"diversity": 0.0, "pairwise_distances": [], "num_comparisons": 0}

    def calculate_drug_likeness(self, smiles_list: List[str]) -> Dict:
        """Calculate drug-likeness using Lipinski's Rule of Five"""
        drug_like_count = 0
        properties = {
            "molecular_weights": [],
            "log_p_values": [],
            "hbd_counts": [],
            "hba_counts": [],
            "rotatable_bonds": []
        }

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mw = Descriptors.MolWt(mol)
                    logp = Crippen.MolLogP(mol)
                    hbd = Lipinski.NumHDonors(mol)
                    hba = Lipinski.NumHAcceptors(mol)
                    rotatable = Descriptors.NumRotatableBonds(mol)

                    properties["molecular_weights"].append(mw)
                    properties["log_p_values"].append(logp)
                    properties["hbd_counts"].append(hbd)
                    properties["hba_counts"].append(hba)
                    properties["rotatable_bonds"].append(rotatable)

                    # Lipinski's Rule of Five
                    if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                        drug_like_count += 1
            except:
                continue

        total_valid = len(properties["molecular_weights"])
        drug_likeness_ratio = drug_like_count / total_valid if total_valid > 0 else 0

        return {
            "drug_likeness": drug_likeness_ratio,
            "drug_like_count": drug_like_count,
            "total_evaluated": total_valid,
            "properties": properties
        }

    def calculate_scaffold_diversity(self, smiles_list: List[str]) -> Dict:
        """Calculate scaffold diversity using Murcko scaffolds"""
        scaffolds = []

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    if scaffold:
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        scaffolds.append(scaffold_smiles)
            except:
                continue

        total_molecules = len(scaffolds)
        unique_scaffolds = len(set(scaffolds))
        scaffold_diversity = unique_scaffolds / total_molecules if total_molecules > 0 else 0

        return {
            "scaffold_diversity": scaffold_diversity,
            "unique_scaffolds": unique_scaffolds,
            "total_molecules": total_molecules,
            "scaffold_distribution": Counter(scaffolds)
        }

    def comprehensive_evaluation(self, generated_smiles: List[str], reference_smiles: List[str] = None) -> Dict:
        """Run comprehensive evaluation of generated molecules with enhanced novelty"""
        results = {}

        # Basic validity and uniqueness
        results["validity"] = self.calculate_validity(generated_smiles)
        results["uniqueness"] = self.calculate_uniqueness(generated_smiles)

        # Get valid unique molecules for further analysis
        valid_unique_smiles = []
        canonical_set = set()
        for smiles in generated_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    if canonical not in canonical_set:
                        canonical_set.add(canonical)
                        valid_unique_smiles.append(canonical)
            except:
                continue

        # Traditional novelty (if reference provided)
        if reference_smiles and reference_smiles[0]:  # Check if not empty
            results["novelty"] = self.calculate_novelty(valid_unique_smiles, reference_smiles)
        else:
            results["novelty"] = {"novelty": 1.0, "novel_count": len(valid_unique_smiles),
                                  "total_generated": len(valid_unique_smiles), "overlap_count": 0}

        # Enhanced novelty against ZINC250k
        enhanced_novelty_result = self.calculate_enhanced_novelty(valid_unique_smiles, "zinc250k")
        results["enhanced_novelty"] = enhanced_novelty_result

        # Diversity and drug-likeness
        results["diversity"] = self.calculate_diversity(valid_unique_smiles)
        results["drug_likeness"] = self.calculate_drug_likeness(valid_unique_smiles)
        results["scaffold_diversity"] = self.calculate_scaffold_diversity(valid_unique_smiles)

        # Summary metrics (enhanced with ZINC novelty)
        zinc_novelty = enhanced_novelty_result.get("enhanced_novelty", 0.0)
        results["summary"] = {
            "total_generated": len(generated_smiles),
            "valid_unique_count": len(valid_unique_smiles),
            "overall_success_rate": len(valid_unique_smiles) / len(generated_smiles) if generated_smiles else 0,
            "zinc_novelty_rate": zinc_novelty,
            "zinc_novel_count": enhanced_novelty_result.get("novel_count", 0),
            "zinc_known_count": enhanced_novelty_result.get("known_count", 0)
        }

        return results

    # FIXED: Overlap calculation methods in MolecularMetrics class
    def calculate_top_n_overlap(self, list1: List[str], list2: List[str], n: int = 5) -> Dict:
        """
        FIXED: Calculate overlap between top-N molecules from two different approaches
        """
        # FIXED: Handle empty lists
        if not list1 or not list2:
            return {
                "top_n": n,
                "list1_top_n": [],
                "list2_top_n": [],
                "overlap_molecules": [],
                "overlap_count": 0,
                "overlap_percentage": 0.0,
                "jaccard_similarity": 0.0,
                "list1_count": 0,
                "list2_count": 0,
                "max_possible_overlap": 0,
                "list1_internal_diversity": 0,
                "list2_internal_diversity": 0,
                "list1_duplicates": 0,
                "list2_duplicates": 0,
                "error": "One or both input lists are empty"
            }

        # FIXED: Get top-N from each list with proper filtering
        def get_top_n_unique(smiles_list, n):
            seen = set()
            unique_list = []
            for smiles in smiles_list:
                # FIXED: Filter out None, empty strings, and invalid entries
                if smiles and isinstance(smiles, str) and smiles.strip() and smiles not in seen:
                    seen.add(smiles.strip())
                    unique_list.append(smiles.strip())
                    if len(unique_list) >= n:
                        break
            return unique_list

        top_n_list1 = get_top_n_unique(list1, n)
        top_n_list2 = get_top_n_unique(list2, n)

        # FIXED: Handle case where we don't have enough molecules
        if not top_n_list1 or not top_n_list2:
            return {
                "top_n": n,
                "list1_top_n": top_n_list1,
                "list2_top_n": top_n_list2,
                "overlap_molecules": [],
                "overlap_count": 0,
                "overlap_percentage": 0.0,
                "jaccard_similarity": 0.0,
                "list1_count": len(top_n_list1),
                "list2_count": len(top_n_list2),
                "max_possible_overlap": 0,
                "list1_internal_diversity": len(set(top_n_list1)),
                "list2_internal_diversity": len(set(top_n_list2)),
                "list1_duplicates": len(top_n_list1) - len(set(top_n_list1)),
                "list2_duplicates": len(top_n_list2) - len(set(top_n_list2)),
                "warning": "Insufficient molecules for meaningful overlap analysis"
            }

        # Calculate overlap between the two lists
        set1 = set(top_n_list1)
        set2 = set(top_n_list2)

        overlap = set1.intersection(set2)
        overlap_count = len(overlap)

        # Calculate overlap percentage
        max_possible_overlap = min(len(top_n_list1), len(top_n_list2))
        overlap_percentage = overlap_count / max_possible_overlap if max_possible_overlap > 0 else 0

        # Jaccard similarity (intersection over union)
        union = set1.union(set2)
        jaccard_similarity = len(overlap) / len(union) if len(union) > 0 else 0

        # Calculate internal diversity within each list
        list1_internal_diversity = len(set(top_n_list1))
        list2_internal_diversity = len(set(top_n_list2))

        return {
            "top_n": n,
            "list1_top_n": top_n_list1,
            "list2_top_n": top_n_list2,
            "overlap_molecules": list(overlap),
            "overlap_count": overlap_count,
            "overlap_percentage": overlap_percentage,
            "jaccard_similarity": jaccard_similarity,
            "list1_count": len(top_n_list1),
            "list2_count": len(top_n_list2),
            "max_possible_overlap": max_possible_overlap,
            "list1_internal_diversity": list1_internal_diversity,
            "list2_internal_diversity": list2_internal_diversity,
            "list1_duplicates": len(top_n_list1) - list1_internal_diversity,
            "list2_duplicates": len(top_n_list2) - list2_internal_diversity
        }

    def calculate_positional_overlap(self, list1: List[str], list2: List[str], n: int = 5) -> Dict:
        """
        FIXED: Calculate position-aware overlap (how many molecules appear in same positions)
        """
        # FIXED: Handle empty lists
        if not list1 or not list2:
            return {
                "positional_matches": 0,
                "total_positions_compared": 0,
                "positional_overlap_rate": 0.0,
                "position_details": [],
                "error": "One or both input lists are empty"
            }

        # FIXED: Get top-n with proper bounds checking
        top_n_list1 = [mol for mol in list1[:n] if mol and isinstance(mol, str) and mol.strip()]
        top_n_list2 = [mol for mol in list2[:n] if mol and isinstance(mol, str) and mol.strip()]

        if not top_n_list1 or not top_n_list2:
            return {
                "positional_matches": 0,
                "total_positions_compared": 0,
                "positional_overlap_rate": 0.0,
                "position_details": [],
                "warning": "Insufficient valid molecules for positional analysis"
            }

        positional_matches = 0
        position_details = []

        max_positions = min(len(top_n_list1), len(top_n_list2))

        for i in range(max_positions):
            if top_n_list1[i] == top_n_list2[i]:
                positional_matches += 1
                position_details.append({
                    "position": i + 1,
                    "molecule": top_n_list1[i],
                    "match": True
                })
            else:
                position_details.append({
                    "position": i + 1,
                    "list1_molecule": top_n_list1[i],
                    "list2_molecule": top_n_list2[i],
                    "match": False
                })

        positional_overlap_rate = positional_matches / max_positions if max_positions > 0 else 0

        return {
            "positional_matches": positional_matches,
            "total_positions_compared": max_positions,
            "positional_overlap_rate": positional_overlap_rate,
            "position_details": position_details
        }


class MetricsVisualizer:
    """Create clear, informative visualizations for molecular generation metrics"""

    def __init__(self, figsize=(16, 12)):
        self.figsize = figsize

    def plot_comprehensive_metrics(self, metrics_dict: Dict, title: str = "Molecular Generation Metrics"):
        """Create comprehensive visualization of all metrics with molecule tracking and enhanced novelty"""
        fig, axes = plt.subplots(3, 3, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 1. Molecule Generation Overview
        ax1 = axes[0, 0]
        if 'molecule_tracking' in metrics_dict:
            tracking = metrics_dict['molecule_tracking']
            categories = ['Generated', 'Valid', 'Invalid']
            values = [
                tracking.get('total_generated', 0),
                tracking.get('total_valid', 0),
                tracking.get('total_invalid', 0)
            ]
            colors = ['#4682B4', '#32CD32', '#DC143C']
            bars = ax1.bar(categories, values, color=colors, alpha=0.8)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{value}', ha='center', va='bottom', fontweight='bold')

            ax1.set_ylabel('Count')
            ax1.set_title('Molecule Generation Overview')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No tracking data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Molecule Generation Overview')

        # 2. Quality Metrics (Enhanced with ZINC Novelty)
        ax2 = axes[0, 1]
        metrics_names = ['Validity', 'Uniqueness', 'Drug-likeness', 'ZINC Novelty']
        metrics_values = [
            metrics_dict.get('validity', {}).get('validity', 0),
            metrics_dict.get('uniqueness', {}).get('uniqueness', 0),
            metrics_dict.get('drug_likeness', {}).get('drug_likeness', 0),
            metrics_dict.get('enhanced_novelty', {}).get('enhanced_novelty', 0)
        ]
        colors = ['#2E8B57', '#4682B4', '#DC143C', '#FF8C00']
        bars = ax2.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
        ax2.set_ylabel('Ratio')
        ax2.set_title('Quality Metrics (with ZINC Novelty)')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # Add value labels
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 3. Novelty Comparison (Traditional vs ZINC)
        ax3 = axes[0, 2]
        novelty_types = ['Traditional\nNovelty', 'ZINC250k\nNovelty']
        novelty_values = [
            metrics_dict.get('novelty', {}).get('novelty', 0),
            metrics_dict.get('enhanced_novelty', {}).get('enhanced_novelty', 0)
        ]
        colors = ['#9370DB', '#FF8C00']

        bars = ax3.bar(novelty_types, novelty_values, color=colors, alpha=0.8)
        ax3.set_ylabel('Novelty Rate')
        ax3.set_title('Novelty Comparison')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)

        # Add value labels and dataset info
        for bar, value in zip(bars, novelty_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Add ZINC dataset size info
        zinc_size = metrics_dict.get('enhanced_novelty', {}).get('reference_size', 0)
        if zinc_size > 0:
            ax3.text(0.5, -0.15, f'ZINC250k size: {zinc_size:,} molecules',
                     transform=ax3.transAxes, ha='center', fontsize=8, style='italic')

        # 4. Molecular Weight Distribution
        ax4 = axes[1, 0]
        if 'drug_likeness' in metrics_dict and 'properties' in metrics_dict['drug_likeness']:
            mw_values = metrics_dict['drug_likeness']['properties']['molecular_weights']
            if mw_values:
                ax4.hist(mw_values, bins=min(15, len(mw_values)), alpha=0.7, color='#4682B4',
                         edgecolor='black', linewidth=1)
                ax4.axvline(x=500, color='red', linestyle='--', linewidth=2, label='Lipinski limit (500)')
                ax4.set_xlabel('Molecular Weight (Da)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Molecular Weight Distribution')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No MW data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Molecular Weight Distribution')

        # 5. LogP Distribution
        ax5 = axes[1, 1]
        if 'drug_likeness' in metrics_dict and 'properties' in metrics_dict['drug_likeness']:
            logp_values = metrics_dict['drug_likeness']['properties']['log_p_values']
            if logp_values:
                ax5.hist(logp_values, bins=min(15, len(logp_values)), alpha=0.7, color='#32CD32',
                         edgecolor='black', linewidth=1)
                ax5.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Lipinski limit (5)')
                ax5.set_xlabel('LogP')
                ax5.set_ylabel('Frequency')
                ax5.set_title('LogP Distribution')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No LogP data', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('LogP Distribution')

        # 6. Diversity Analysis
        ax6 = axes[1, 2]
        if 'diversity' in metrics_dict and 'pairwise_distances' in metrics_dict['diversity']:
            distances = metrics_dict['diversity']['pairwise_distances']
            if distances:
                ax6.hist(distances, bins=min(15, len(distances)), alpha=0.7, color='#FF6347',
                         edgecolor='black', linewidth=1)
                mean_div = np.mean(distances)
                ax6.axvline(x=mean_div, color='blue', linestyle='--', linewidth=2,
                            label=f'Mean: {mean_div:.3f}')
                ax6.set_xlabel('Tanimoto Distance')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Molecular Diversity')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No diversity data', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Molecular Diversity')

        # 7. Scaffold Analysis
        ax7 = axes[2, 0]
        if 'scaffold_diversity' in metrics_dict:
            scaffold_div = metrics_dict['scaffold_diversity']['scaffold_diversity']
            total_scaffolds = metrics_dict['scaffold_diversity']['total_molecules']
            unique_scaffolds = metrics_dict['scaffold_diversity']['unique_scaffolds']

            if total_scaffolds > 0:
                categories = ['Total\nMolecules', 'Unique\nScaffolds']
                values = [total_scaffolds, unique_scaffolds]
                colors = ['#4682B4', '#32CD32']

                bars = ax7.bar(categories, values, color=colors, alpha=0.8)
                ax7.set_ylabel('Count')
                ax7.set_title(f'Scaffold Analysis\n(Diversity: {scaffold_div:.3f})')
                ax7.grid(True, alpha=0.3)

                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax7.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                             f'{value}', ha='center', va='bottom', fontweight='bold')
            else:
                ax7.text(0.5, 0.5, 'No scaffold data', ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title('Scaffold Analysis')

        # 8. ZINC Novelty Breakdown
        ax8 = axes[2, 1]
        enhanced_novelty = metrics_dict.get('enhanced_novelty', {})
        if enhanced_novelty.get('status') == 'success':
            categories = ['Novel\n(New)', 'Known\n(ZINC)', 'Invalid']
            values = [
                enhanced_novelty.get('novel_count', 0),
                enhanced_novelty.get('known_count', 0),
                enhanced_novelty.get('invalid_generated', 0)
            ]
            colors = ['#32CD32', '#FF6347', '#808080']

            bars = ax8.bar(categories, values, color=colors, alpha=0.8)
            ax8.set_ylabel('Count')
            ax8.set_title('ZINC250k Novelty Breakdown')
            ax8.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{value}', ha='center', va='bottom', fontweight='bold')
        else:
            error_msg = enhanced_novelty.get('error', 'ZINC data unavailable')
            ax8.text(0.5, 0.5, f'ZINC Error:\n{error_msg[:50]}...', ha='center', va='center',
                     transform=ax8.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))
            ax8.set_title('ZINC250k Novelty Breakdown')

        # 9. Enhanced Summary Statistics
        ax9 = axes[2, 2]
        ax9.axis('off')

        summary_text = "ENHANCED SUMMARY:\n\n"
        if 'molecule_tracking' in metrics_dict:
            tracking = metrics_dict['molecule_tracking']
            summary_text += f"Generated: {tracking.get('total_generated', 0)}\n"
            summary_text += f"Valid: {tracking.get('total_valid', 0)}\n"
            summary_text += f"Invalid: {tracking.get('total_invalid', 0)}\n\n"

        if 'summary' in metrics_dict:
            summary = metrics_dict['summary']
            summary_text += f"Success Rate: {summary.get('overall_success_rate', 0):.3f}\n"
            summary_text += f"ZINC Novelty: {summary.get('zinc_novelty_rate', 0):.3f}\n"
            summary_text += f"ZINC Novel: {summary.get('zinc_novel_count', 0)}\n"
            summary_text += f"ZINC Known: {summary.get('zinc_known_count', 0)}\n\n"

        summary_text += f"Validity: {metrics_dict.get('validity', {}).get('validity', 0):.3f}\n"
        summary_text += f"Uniqueness: {metrics_dict.get('uniqueness', {}).get('uniqueness', 0):.3f}\n"
        summary_text += f"Diversity: {metrics_dict.get('diversity', {}).get('diversity', 0):.3f}\n"
        summary_text += f"Drug-likeness: {metrics_dict.get('drug_likeness', {}).get('drug_likeness', 0):.3f}"

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax9.set_title('Enhanced Statistics')

        plt.tight_layout()
        return fig

    def plot_pipeline_comparison(self, single_shot_metrics: Dict, iterative_metrics: Dict):
        """Compare single-shot vs iterative pipeline metrics with enhanced novelty"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pipeline Comparison: Single-shot vs Iterative (Enhanced)', fontsize=16, fontweight='bold')

        # 1. Molecule Generation Comparison
        ax1 = axes[0, 0]
        categories = ['Generated', 'Valid', 'Invalid']

        ss_tracking = single_shot_metrics.get('molecule_tracking', {})
        it_tracking = iterative_metrics.get('molecule_tracking', {})

        ss_values = [
            ss_tracking.get('total_generated', 0),
            ss_tracking.get('total_valid', 0),
            ss_tracking.get('total_invalid', 0)
        ]

        it_values = [
            it_tracking.get('total_generated', 0),
            it_tracking.get('total_valid', 0),
            it_tracking.get('total_invalid', 0)
        ]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, ss_values, width, label='Single-shot', alpha=0.8, color='#4682B4')
        bars2 = ax1.bar(x + width / 2, it_values, width, label='Iterative', alpha=0.8, color='#32CD32')

        ax1.set_xlabel('Molecule Categories')
        ax1.set_ylabel('Count')
        ax1.set_title('Molecule Generation Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # 2. Enhanced Quality Metrics Comparison (with ZINC Novelty)
        ax2 = axes[0, 1]
        quality_metrics = ['Validity', 'Uniqueness', 'Drug-likeness', 'ZINC Novelty']

        ss_quality = [
            single_shot_metrics.get('validity', {}).get('validity', 0),
            single_shot_metrics.get('uniqueness', {}).get('uniqueness', 0),
            single_shot_metrics.get('drug_likeness', {}).get('drug_likeness', 0),
            single_shot_metrics.get('enhanced_novelty', {}).get('enhanced_novelty', 0)
        ]

        it_quality = [
            iterative_metrics.get('validity', {}).get('validity', 0),
            iterative_metrics.get('uniqueness', {}).get('uniqueness', 0),
            iterative_metrics.get('drug_likeness', {}).get('drug_likeness', 0),
            iterative_metrics.get('enhanced_novelty', {}).get('enhanced_novelty', 0)
        ]

        x = np.arange(len(quality_metrics))
        bars1 = ax2.bar(x - width / 2, ss_quality, width, label='Single-shot', alpha=0.8, color='#4682B4')
        bars2 = ax2.bar(x + width / 2, it_quality, width, label='Iterative', alpha=0.8, color='#32CD32')

        ax2.set_xlabel('Quality Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Enhanced Quality Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(quality_metrics, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 3. ZINC Novelty Detailed Comparison
        ax3 = axes[1, 0]
        novelty_categories = ['Novel\n(ZINC)', 'Known\n(ZINC)']

        ss_enhanced = single_shot_metrics.get('enhanced_novelty', {})
        it_enhanced = iterative_metrics.get('enhanced_novelty', {})

        ss_novelty_values = [
            ss_enhanced.get('novel_count', 0),
            ss_enhanced.get('known_count', 0)
        ]

        it_novelty_values = [
            it_enhanced.get('novel_count', 0),
            it_enhanced.get('known_count', 0)
        ]

        x = np.arange(len(novelty_categories))
        bars1 = ax3.bar(x - width / 2, ss_novelty_values, width, label='Single-shot', alpha=0.8, color='#4682B4')
        bars2 = ax3.bar(x + width / 2, it_novelty_values, width, label='Iterative', alpha=0.8, color='#32CD32')

        ax3.set_xlabel('ZINC250k Categories')
        ax3.set_ylabel('Count')
        ax3.set_title('ZINC250k Novelty Detailed Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(novelty_categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{int(height)}', ha='center', va='bottom', fontsize=9)

        # 4. Enhanced Summary Comparison
        ax4 = axes[1, 1]
        ax4.axis('off')

        comparison_text = "ENHANCED COMPARISON SUMMARY:\n\n"

        # ZINC novelty rates
        ss_zinc_rate = ss_enhanced.get('enhanced_novelty', 0)
        it_zinc_rate = it_enhanced.get('enhanced_novelty', 0)

        comparison_text += f"ZINC250k Novelty Rates:\n"
        comparison_text += f"Single-shot: {ss_zinc_rate:.3f}\n"
        comparison_text += f"Iterative:   {it_zinc_rate:.3f}\n\n"

        # Novel molecule counts
        comparison_text += f"Novel Molecules (ZINC):\n"
        comparison_text += f"Single-shot: {ss_enhanced.get('novel_count', 0)}\n"
        comparison_text += f"Iterative:   {it_enhanced.get('novel_count', 0)}\n\n"

        # Success rates
        ss_success = single_shot_metrics.get('summary', {}).get('overall_success_rate', 0)
        it_success = iterative_metrics.get('summary', {}).get('overall_success_rate', 0)

        comparison_text += f"Overall Success Rates:\n"
        comparison_text += f"Single-shot: {ss_success:.3f}\n"
        comparison_text += f"Iterative:   {it_success:.3f}\n\n"

        # ZINC dataset info
        zinc_size = ss_enhanced.get('reference_size', it_enhanced.get('reference_size', 0))
        if zinc_size > 0:
            comparison_text += f"Reference: ZINC250k ({zinc_size:,} molecules)"

        ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax4.set_title('Enhanced Summary Comparison')

        plt.tight_layout()
        return fig

    def plot_top_n_overlap_analysis(self, overlap_data: Dict, title: str = "Top-N Overlap Analysis"):
        """FIXED: Create comprehensive visualization for top-N overlap analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # FIXED: Check for errors in overlap data
        if overlap_data.get("error") or overlap_data.get("warning"):
            error_msg = overlap_data.get("error", overlap_data.get("warning", "Unknown error"))
            fig.text(0.5, 0.5, f"⚠️ Overlap Analysis Error:\n{error_msg}",
                     ha='center', va='center', fontsize=14,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            return fig

        n = overlap_data['top_n']

        # FIXED: Handle case with no overlap data
        if overlap_data['overlap_count'] == 0 and overlap_data['list1_count'] == 0 and overlap_data['list2_count'] == 0:
            fig.text(0.5, 0.5, "⚠️ No molecules available for overlap analysis",
                     ha='center', va='center', fontsize=14,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            return fig

        # 1. FIXED: Overlap Summary Bar Chart
        ax1 = axes[0, 0]
        categories = ['Overlap', 'Single-shot\nUnique', 'Iterative\nUnique']
        overlap_count = overlap_data['overlap_count']
        list1_unique = max(0, overlap_data['list1_count'] - overlap_count)
        list2_unique = max(0, overlap_data['list2_count'] - overlap_count)

        values = [overlap_count, list1_unique, list2_unique]
        colors = ['#32CD32', '#4682B4', '#FF6347']

        bars = ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_ylabel('Count')
        ax1.set_title(f'Top-{n} Molecule Distribution')
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{value}', ha='center', va='bottom', fontweight='bold')

        # FIXED: Add data quality indicator
        ax1.text(0.5, -0.15, f'Data: {overlap_data["list1_count"]} vs {overlap_data["list2_count"]} molecules',
                 transform=ax1.transAxes, ha='center', fontsize=10, style='italic')

        # 2. Overlap Metrics
        ax2 = axes[0, 1]
        metrics = ['Overlap %', 'Jaccard\nSimilarity']
        metric_values = [
            overlap_data['overlap_percentage'] * 100,
            overlap_data['jaccard_similarity'] * 100
        ]

        bars = ax2.bar(metrics, metric_values, color=['#9370DB', '#20B2AA'], alpha=0.8)
        ax2.set_ylabel('Percentage')
        ax2.set_title('Overlap Metrics')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Internal Diversity Analysis
        ax3 = axes[0, 2]
        diversity_categories = ['Single-shot\nDiversity', 'Iterative\nDiversity']
        diversity_values = [
            overlap_data['list1_internal_diversity'],
            overlap_data['list2_internal_diversity']
        ]
        max_values = [overlap_data['list1_count'], overlap_data['list2_count']]

        bars1 = ax3.bar(diversity_categories, diversity_values, color=['#4682B4', '#FF6347'], alpha=0.8, label='Unique')
        bars2 = ax3.bar(diversity_categories,
                        [overlap_data['list1_duplicates'], overlap_data['list2_duplicates']],
                        bottom=diversity_values, color=['#87CEEB', '#FFA07A'], alpha=0.8, label='Duplicates')

        ax3.set_ylabel('Count')
        ax3.set_title(f'Internal Diversity (Top-{n})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, diversity_values)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height / 2,
                     f'{value}', ha='center', va='center', fontweight='bold', color='white')

        # 4. Venn Diagram Style Visualization
        ax4 = axes[1, 0]
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 10)

        # Draw circles (simplified Venn diagram)
        circle1 = plt.Circle((3, 5), 2.5, alpha=0.3, color='blue', label='Single-shot')
        circle2 = plt.Circle((7, 5), 2.5, alpha=0.3, color='red', label='Iterative')

        ax4.add_patch(circle1)
        ax4.add_patch(circle2)

        # Add text annotations
        ax4.text(2, 5, f'{list1_unique}', ha='center', va='center', fontweight='bold', fontsize=12)
        ax4.text(8, 5, f'{list2_unique}', ha='center', va='center', fontweight='bold', fontsize=12)
        ax4.text(5, 5, f'{overlap_count}', ha='center', va='center', fontweight='bold', fontsize=14)

        ax4.set_title('Overlap Visualization')
        ax4.legend()
        ax4.set_aspect('equal')
        ax4.axis('off')

        # 5. Overlapping Molecules Display
        ax5 = axes[1, 1]
        ax5.axis('off')

        overlap_text = f"OVERLAPPING MOLECULES (Top-{n}):\n\n"
        overlap_mols = overlap_data['overlap_molecules']

        if overlap_mols:
            for i, mol in enumerate(overlap_mols, 1):
                overlap_text += f"{i}. {mol[:45]}{'...' if len(mol) > 45 else ''}\n"
        else:
            overlap_text += "❌ No overlapping molecules found\n"

        overlap_text += f"\nOVERLAP STATISTICS:\n"
        overlap_text += f"• Total Overlap: {overlap_count}/{min(overlap_data['list1_count'], overlap_data['list2_count'])}\n"
        overlap_text += f"• Overlap Rate: {overlap_data['overlap_percentage'] * 100:.1f}%\n"
        overlap_text += f"• Jaccard Similarity: {overlap_data['jaccard_similarity'] * 100:.1f}%"

        ax5.text(0.05, 0.95, overlap_text, transform=ax5.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax5.set_title('Overlapping Molecules')

        # 6. Unique Molecules Comparison
        ax6 = axes[1, 2]
        ax6.axis('off')

        comparison_text = f"UNIQUE MOLECULES COMPARISON (Top-{n}):\n\n"

        # Show all single-shot unique molecules (up to top N)
        comparison_text += f"SINGLE-SHOT UNIQUE:\n"
        list1_only = [mol for mol in overlap_data['list1_top_n'] if mol not in overlap_mols]
        if list1_only:
            for i, mol in enumerate(list1_only[:n], 1):  # Show up to N molecules
                comparison_text += f"{i}. {mol[:40]}{'...' if len(mol) > 40 else ''}\n"
        else:
            comparison_text += "None (all molecules overlap)\n"

        comparison_text += f"\nITERATIVE UNIQUE:\n"
        list2_only = [mol for mol in overlap_data['list2_top_n'] if mol not in overlap_mols]
        if list2_only:
            for i, mol in enumerate(list2_only[:n], 1):  # Show up to N molecules
                comparison_text += f"{i}. {mol[:40]}{'...' if len(mol) > 40 else ''}\n"
        else:
            comparison_text += "None (all molecules overlap)\n"

        ax6.text(0.05, 0.95, comparison_text, transform=ax6.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        ax6.set_title('Unique Molecules Details')

        plt.tight_layout()
        return fig