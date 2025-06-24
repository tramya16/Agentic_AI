from crew.coordinator import run_molecular_pipeline

if __name__ == "__main__":
    user_input = "Design a molecule similar to albuterol while preserving key functional groups."

    print("Running full pipeline...\n")
    results = run_molecular_pipeline(user_input)

    print("\n===== Final Results =====")
    for key, val in results.items():
        print(f"\n--- {key.upper()} ---")
        print(val)
