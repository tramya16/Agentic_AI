# run_experiment.py

from overlap_experiment import OverlapExperiment
from queries import get_query_list


def main():
    """Main execution function"""
    experiment = OverlapExperiment()

    # Configuration
    RUNS_PER_QUERY = 5
    TOP_N = 5

    # Choose queries to run
    # Option 1: Run all queries
    # query_names = get_query_list()

    # Option 2: Run specific queries for testing
    query_names = [
        "albuterol_similarity",
        "amlodipine_mpo",
        "celecoxib_rediscovery",
        "drd2",
        "fexofenadine_mpo"
    ]

    print(f"Running overlap experiment:")
    print(f"- Queries: {len(query_names)}")
    print(f"- Runs per query: {RUNS_PER_QUERY}")
    print(f"- Top N molecules: {TOP_N}")
    print(f"- Total pipeline runs: {len(query_names) * RUNS_PER_QUERY * 2}")  # 2 pipelines

    try:
        results, analyses = experiment.run_full_experiment(
            query_names=query_names,
            runs=RUNS_PER_QUERY,
            top_n=TOP_N
        )

        print(f"\n✓ Experiment completed successfully!")
        print(f"✓ Results saved to: {experiment.results_dir}")
        print(f"✓ Check summary_table_*.txt for readable results")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()