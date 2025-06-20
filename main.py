from crew.crew_setup import run_molecular_parser

if __name__ == "__main__":
    user_input = "Generate a new molecule similar to aspirin CC(=O)OC1=CC=CC=C1C(=O)O with low toxicity and high solubility"

    print("=" * 80)
    print("Starting Molecular Design Pipeline")
    print("=" * 80)
    print(f"User Request: {user_input}\n")

    try:
        result = run_molecular_parser(user_input)
        print("\n" + "=" * 80)
        print("Parsing Result:")
        print("=" * 80)
        print(result)
    except Exception as e:
        print(f"Pipeline Error: {e}")
        import traceback

        traceback.print_exc()