import requests

PUBCHEM_CID = "33613"
URL = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{PUBCHEM_CID}/property/IsomericSMILES/TXT"

def fetch_smiles() -> str:
    resp = requests.get(URL)
    resp.raise_for_status()
    return resp.text.strip()

if __name__ == "__main__":
    smiles = fetch_smiles()
    with open("data/amoxicillin.smi", "w") as f:
        f.write(smiles + "\n")
    print("Fetched Amoxicillin SMILES:", smiles)