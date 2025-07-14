from crew.coordinator import run_molecular_pipeline

if __name__ == "__main__":
    # grab a prompt from the CLI, a GUI, or hard-code one for testing
    user_input = "Generate me a molecule that inhibits EGFR"
    results = run_molecular_pipeline(user_input)
    # pretty-print or otherwise use the results
    import pprint; pprint.pprint(results)