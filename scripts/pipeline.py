import csv
from pathlib import Path
from dotenv import load_dotenv
from rdkit import Chem

from agents.generator_agent import GeneratorAgent
from agents.validator_agent import ValidatorAgent
from chemtools.scorer import get_all_scores

# ————— Configuration —————
# Load env vars (e.g. OPENAI_API_KEY) from project root .env
load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ROOT = Path(__file__).parent.parent
BASE_SMILES_PATH = PROJECT_ROOT / "data" / "amoxicillin.smi"
LOG_CSV_PATH   = PROJECT_ROOT / "data" / "pipeline_log.csv"

OBJECTIVE = "reduce toxicity and increase solubility"

# ————— Helpers —————
def load_base_smiles(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"No SMILES at {path}")
    return path.read_text().strip()

def append_log(row: dict):
    """Append a row to the CSV log, writing headers if file new."""
    write_header = not LOG_CSV_PATH.exists()
    with open(LOG_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ————— Main Loop —————
def main():
    base_smiles = load_base_smiles(BASE_SMILES_PATH)
    print("Base SMILES:", base_smiles)

    gen_agent = GeneratorAgent()
    val_agent = ValidatorAgent()

    # Run one generation/validation cycle (could be looped or crew-managed)
    generated = gen_agent.generate(base_smiles, OBJECTIVE)
    print("Generated SMILES:", generated)

    is_valid = val_agent.validate(generated)
    canonical = ""
    scores = {}
    error = ""

    if is_valid:
        # Standardize + score
        canonical = val_agent.sanitize(generated)
        scores = get_all_scores(canonical)
        print("Valid. Canonical SMILES:", canonical)
        print("Scores:", scores)
    else:
        error = "Validation failed"
        print("Invalid SMILES")

    # Log everything
    log_row = {
        "base_smiles":      base_smiles,
        "generated_smiles": generated,
        "is_valid":         is_valid,
        "canonical":        canonical,
        "logp":             scores.get("LogP", ""),
        "molwt":            scores.get("MolWt", ""),
        "tpsa":             scores.get("TPSA", ""),
        "error":            error,
    }
    append_log(log_row)

if __name__ == "__main__":
    main()
