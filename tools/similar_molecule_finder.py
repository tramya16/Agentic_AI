# tools/similar_molecule_finder.py
import json
import requests
from urllib.parse import quote
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class SimilarMoleculeInput(BaseModel):
    smiles: str = Field(..., description="Reference SMILES string")
    similarity_threshold: float = Field(
        0.7,
        description="Minimum similarity score (0-1)",
        ge=0.0,
        le=1.0
    )
    max_results: int = Field(
        5,
        description="Maximum number of results to return",
        gt=0,
        le=50
    )

class SimilarMoleculeFinderTool(BaseTool):
    name: str = "similar_molecule_finder"
    description: str = (
        "Find Structurally similar molecule using PubChem's fastsimilarity_2d endpoint. "
        "Returns top hits with Tanimoto scores. Includes debug output."
    )
    args_schema: Type[BaseModel] = SimilarMoleculeInput

    def _run(self, smiles: str, similarity_threshold: float = 0.7, max_results: int = 5) -> str:
        try:
            print(f"[DEBUG] Input SMILES: {smiles}")  # debug
            # Generate reference fingerprint
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print("[DEBUG] Invalid SMILES provided.")
                return json.dumps({"status": "error", "error": "Invalid SMILES"})
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

            # Query PubChem fastsimilarity_2d endpoint
            encoded = quote(smiles, safe='')  # percent-encode special chars
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                f"fastsimilarity_2d/smiles/{encoded}/cids/JSON"
            )
            params = {
                "Threshold": int(similarity_threshold * 100),
                "MaxRecords": max_results
            }
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            cids = data.get("IdentifierList", {}).get("CID", [])

            results = []
            # For each CID, fetch SMILES + name and compute exact similarity
            for cid in cids:
                prop_url = (
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                    f"cid/{cid}/property/CanonicalSMILES,IUPACName/JSON"
                )
                p = requests.get(prop_url, timeout=10).json()
                props = p.get("PropertyTable", {}).get("Properties", [{}])[0]
                hit_smiles = props.get("ConnectivitySMILES")
                name = props.get("IUPACName", "N/A")
                if not hit_smiles:
                    continue
                hit_mol = Chem.MolFromSmiles(hit_smiles)
                if not hit_mol:
                    continue
                hit_fp = AllChem.GetMorganFingerprintAsBitVect(hit_mol, 2, nBits=2048)
                score = DataStructs.TanimotoSimilarity(ref_fp, hit_fp)
                results.append({
                    "cid": cid,
                    "smiles": hit_smiles,
                    "name": name,
                    "similarity": round(score, 4)
                })

            output = {
                "status": "success",
                "reference_smiles": smiles,
                "similarity_threshold": similarity_threshold,
                "results": results
            }
            return json.dumps(output, indent=2)
        except Exception as e:
            print(f"[DEBUG] Exception occurred: {e}")
            return json.dumps({"status": "error", "error": str(e)})
