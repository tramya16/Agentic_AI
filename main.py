import json
from pathlib import Path
from collections import defaultdict
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_smiles_from_file(file_path):
    """Extract SMILES from a single JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    single_shot_smiles = []
    iterative_smiles = []

    # Single-shot extractions
    for run in data.get("single_shot", []):
        smiles = run.get("result", {}).get("valid", [])
        single_shot_smiles.extend(smiles)

    # Iterative extractions
    for run in data.get("iterative", []):
        smiles = run.get("result", {}).get("valid", [])
        iterative_smiles.extend(smiles)

    return single_shot_smiles, iterative_smiles


def extract_all_smiles_from_dir(directory=".", pattern="*json"):
    """Extract SMILES from all JSON files in directory"""
    directory = Path(directory)
    all_single_shot = defaultdict(list)
    all_iterative = defaultdict(list)

    for file_path in directory.glob(pattern):
        if file_path.is_file():
            try:
                single, iterative = extract_smiles_from_file(file_path)
                name = file_path.stem  # e.g., albuterol_similarity_20250722_075459
                all_single_shot[name].extend(single)
                all_iterative[name].extend(iterative)
                print(f"✅ Processed {file_path.name}: {len(single)} single-shot, {len(iterative)} iterative")
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")

    return all_single_shot, all_iterative


def calculate_molecular_properties(smiles):
    """Calculate additional molecular properties for analysis"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    properties = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol)
    }
    return properties


def score_molecules(smiles_list, pipeline_name, query_name, oracle):
    """Enhanced function to validate and score SMILES with additional properties"""
    results = []

    print(f"\nScoring {pipeline_name} molecules from {query_name}:")
    print("-" * 60)

    for i, smi in enumerate(smiles_list, 1):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:  # Valid SMILES
            try:
                score = oracle(smi)
                props = calculate_molecular_properties(smi)

                result = {
                    'SMILES': smi,
                    'Score': score,
                    'Pipeline': pipeline_name,
                    'Query': query_name,
                    'Molecule_ID': f"{query_name}_{pipeline_name}_{i}",
                    **props
                }
                results.append(result)

                print(f"{pipeline_name} {i}: Score = {score:.3f}, MW = {props['MW']:.1f}, LogP = {props['LogP']:.2f}")

            except Exception as e:
                print(f"Error scoring {smi}: {e}")
        else:
            print(f"Invalid SMILES: {smi}")

    return results


def create_comprehensive_visualization(df):
    """Create comprehensive visualization of results"""
    if len(df) == 0:
        print("No valid molecules to visualize!")
        return

    fig = plt.figure(figsize=(16, 12))

    # Score comparison by pipeline
    ax1 = plt.subplot(2, 4, 1)
    sns.boxplot(data=df, x='Pipeline', y='Score', ax=ax1)
    ax1.set_title("Oracle: Albuterol Similarity Scores")
    ax1.set_ylabel("Oracle Similarity Score")
    ax1.tick_params(axis='x', rotation=45)
    # Add mean values as text
    for i, pipeline in enumerate(df['Pipeline'].unique()):
        mean_score = df[df['Pipeline'] == pipeline]['Score'].mean()
        ax1.text(i, mean_score, f'{mean_score:.3f}', ha='center', va='bottom')

    # Individual scores
    ax2 = plt.subplot(2, 4, 2)
    sns.stripplot(data=df, x='Pipeline', y='Score', size=8, ax=ax2)
    ax2.set_title("Individual Oracle Scores")
    ax2.set_ylabel("Oracle Similarity Score")
    ax2.tick_params(axis='x', rotation=45)
    # Add horizontal line at median oracle score
    overall_median = df['Score'].median()
    ax2.axhline(y=overall_median, color='red', linestyle='--', alpha=0.7, label=f'Overall Median: {overall_median:.3f}')
    ax2.legend()

    # Molecular weight distribution
    ax3 = plt.subplot(2, 4, 3)
    sns.violinplot(data=df, x='Pipeline', y='MW', ax=ax3)
    ax3.set_title("Molecular Weight Distribution")
    ax3.set_ylabel("Molecular Weight (Da)")
    ax3.tick_params(axis='x', rotation=45)

    # LogP distribution
    ax4 = plt.subplot(2, 4, 4)
    sns.violinplot(data=df, x='Pipeline', y='LogP', ax=ax4)
    ax4.set_title("LogP Distribution")
    ax4.set_ylabel("LogP")
    ax4.tick_params(axis='x', rotation=45)

    # TPSA distribution
    ax5 = plt.subplot(2, 4, 5)
    sns.violinplot(data=df, x='Pipeline', y='TPSA', ax=ax5)
    ax5.set_title("TPSA Distribution")
    ax5.set_ylabel("TPSA (Ų)")
    ax5.tick_params(axis='x', rotation=45)

    # Score vs LogP scatter
    ax6 = plt.subplot(2, 4, 6)
    for pipeline in df['Pipeline'].unique():
        subset = df[df['Pipeline'] == pipeline]
        ax6.scatter(subset['LogP'], subset['Score'], label=pipeline, alpha=0.7, s=60)
    ax6.set_xlabel("LogP")
    ax6.set_ylabel("Oracle Similarity Score")
    ax6.set_title("Oracle Score vs LogP")
    ax6.legend()
    # Add correlation coefficient
    corr = df['LogP'].corr(df['Score'])
    ax6.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Query comparison (if multiple queries)
    if len(df['Query'].unique()) > 1:
        ax7 = plt.subplot(2, 4, 7)
        sns.boxplot(data=df, x='Query', y='Score', ax=ax7)
        ax7.set_title("Oracle Scores by Query")
        ax7.set_ylabel("Oracle Similarity Score")
        ax7.tick_params(axis='x', rotation=45)
        # Add best scoring query annotation
        query_means = df.groupby('Query')['Score'].mean()
        best_query = query_means.idxmax()
        best_score = query_means.max()
        ax7.text(0.5, 0.95, f'Best Query: {best_query}\nMean Score: {best_score:.3f}',
                 transform=ax7.transAxes, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Molecules per query/pipeline
    ax8 = plt.subplot(2, 4, 8)
    query_pipeline_counts = df.groupby(['Query', 'Pipeline']).size().unstack(fill_value=0)
    query_pipeline_counts.plot(kind='bar', ax=ax8)
    ax8.set_title("Molecules per Query/Pipeline")
    ax8.set_ylabel("Number of Molecules")
    ax8.tick_params(axis='x', rotation=45)
    ax8.legend(title='Pipeline')

    plt.tight_layout()
    plt.show()


def print_comprehensive_analysis(df):
    """Print comprehensive analysis results"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 80)

    print(f"\nTotal molecules analyzed: {len(df)}")
    print(f"Queries: {', '.join(df['Query'].unique())}")
    print(f"Pipelines: {', '.join(df['Pipeline'].unique())}")

    print("\nAlbuterol Similarity Oracle Score Statistics:")
    score_stats = df.groupby(['Query', 'Pipeline'])['Score'].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
    print(score_stats)

    print("\nOverall Pipeline Comparison (Oracle Scores):")
    pipeline_stats = df.groupby('Pipeline')['Score'].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
    print(pipeline_stats)

    # Highlight best performing pipeline
    best_pipeline = pipeline_stats['mean'].idxmax()
    best_mean = pipeline_stats.loc[best_pipeline, 'mean']
    print(f"\n🏆 BEST PERFORMING PIPELINE: {best_pipeline} (Mean Oracle Score: {best_mean:.3f})")

    print("\nMolecular Property Statistics:")
    property_cols = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds']
    for prop in property_cols:
        if prop in df.columns:
            print(f"\n{prop}:")
            prop_stats = df.groupby('Pipeline')[prop].agg(['mean', 'std']).round(2)
            print(prop_stats)

    # Top scoring molecules
    print("\n" + "=" * 80)
    print("🏆 TOP 10 MOLECULES BY ORACLE SIMILARITY SCORE")
    print("=" * 80)
    top_molecules = df.nlargest(10, 'Score')
    for rank, (_, row) in enumerate(top_molecules.iterrows(), 1):
        print(f"\n#{rank} - {row['Molecule_ID']} (Oracle Score: {row['Score']:.3f}):")
        print(f"  SMILES: {row['SMILES']}")
        print(f"  Pipeline: {row['Pipeline']}, Query: {row['Query']}")
        print(f"  MW: {row['MW']:.1f} Da, LogP: {row['LogP']:.2f}")
        print(f"  HBD: {row['HBD']}, HBA: {row['HBA']}")
        print(f"  TPSA: {row['TPSA']:.1f} Ų, RotBonds: {row['RotBonds']}")

    # Oracle score distribution summary
    print(f"\n📊 ORACLE SCORE DISTRIBUTION SUMMARY:")
    print(f"  Total molecules scored: {len(df)}")
    print(f"  Oracle score range: {df['Score'].min():.3f} - {df['Score'].max():.3f}")
    print(f"  Mean oracle score: {df['Score'].mean():.3f}")
    print(f"  Median oracle score: {df['Score'].median():.3f}")
    print(f"  Standard deviation: {df['Score'].std():.3f}")

    # High scoring molecules count
    high_score_threshold = df['Score'].quantile(0.9)  # Top 10%
    high_scorers = df[df['Score'] >= high_score_threshold]
    print(f"  High-scoring molecules (top 10%): {len(high_scorers)} molecules (score ≥ {high_score_threshold:.3f})")


def perform_statistical_analysis(df):
    """Perform statistical analysis if scipy is available"""
    try:
        from scipy import stats

        print(f"\n" + "=" * 80)
        print("STATISTICAL ANALYSIS")
        print("=" * 80)

        # Compare pipelines if both exist
        pipelines = df['Pipeline'].unique()
        if len(pipelines) >= 2:
            for i in range(len(pipelines)):
                for j in range(i + 1, len(pipelines)):
                    pipe1, pipe2 = pipelines[i], pipelines[j]
                    scores1 = df[df['Pipeline'] == pipe1]['Score']
                    scores2 = df[df['Pipeline'] == pipe2]['Score']

                    if len(scores1) > 1 and len(scores2) > 1:
                        t_stat, p_value = stats.ttest_ind(scores1, scores2)
                        print(f"\n{pipe1} vs {pipe2}:")
                        print(f"  t-statistic: {t_stat:.3f}")
                        print(f"  p-value: {p_value:.3f}")
                        print(f"  Significant difference (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

    except ImportError:
        print("\nNote: Install scipy for statistical significance testing")


def main():
    """Main function to run the complete analysis"""
    print("🔬 Starting Comprehensive Molecule Analysis")
    print("=" * 80)

    # Load oracle
    try:
        oracle = Oracle(name="Amlodipine_MPO")  # Changed to albuterol_similarity
        print("✅ Successfully loaded",{oracle},"oracle")
    except Exception as e:
        print(f"❌ Error loading oracle: {e}")
        return

    # Extract SMILES from all JSON files
    print("\n📁 Extracting SMILES from JSON files...")
    single_shot_results, iterative_results = extract_all_smiles_from_dir(directory="scripts/improved_experiment_results/albuterol_similarity_detailed_20250722_074018.json", pattern="*.json")

    if not single_shot_results and not iterative_results:
        print("❌ No SMILES found in JSON files!")
        return

    # Score all molecules
    all_results = []

    # Process single-shot results
    for query_name, smiles_list in single_shot_results.items():
        if smiles_list:
            results = score_molecules(smiles_list, "Single-shot", query_name, oracle)
            all_results.extend(results)

    # Process iterative results
    for query_name, smiles_list in iterative_results.items():
        if smiles_list:
            results = score_molecules(smiles_list, "Iterative", query_name, oracle)
            all_results.extend(results)

    if not all_results:
        print("❌ No valid molecules to analyze!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_comprehensive_visualization(df)

    # Print analysis
    print_comprehensive_analysis(df)

    # Statistical analysis
    perform_statistical_analysis(df)

    print(f"\n🎉 Analysis complete! Evaluated {len(df)} valid molecules from {len(df['Query'].unique())} queries.")

    # Save results to CSV
    output_file = "albuterol_similarity_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()