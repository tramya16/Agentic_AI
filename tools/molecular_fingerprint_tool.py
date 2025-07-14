import json
from crewai.tools import BaseTool
from rdkit import Chem
from rdkit.Chem import AllChem


class MolecularFingerprintTool(BaseTool):
    name: str = "Molecular Fingerprint Generator"
    description: str = "Generate various molecular fingerprints (Morgan, MACCS, RDKit) for similarity analysis."

    def _run(self, smiles: str, fingerprint_type: str = "morgan"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return json.dumps({"error": "Invalid SMILES"})

            fingerprints = {}
            if fingerprint_type in ["morgan", "all"]:
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints["morgan"] = morgan_fp.ToBitString()
            if fingerprint_type in ["maccs", "all"]:
                try:
                    from rdkit.Chem import MACCSkeys
                    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                    fingerprints["maccs"] = maccs_fp.ToBitString()
                except ImportError:
                    # Fallback to Morgan if MACCS not available
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fingerprints["maccs"] = morgan_fp.ToBitString()
            if fingerprint_type in ["rdkit", "all"]:
                rdkit_fp = Chem.RDKFingerprint(mol)
                fingerprints["rdkit"] = rdkit_fp.ToBitString()
            if fingerprint_type in ["topological", "all"]:
                topo_fp = Chem.RDKFingerprint(mol)
                fingerprints["topological"] = topo_fp.ToBitString()

            return json.dumps({
                "smiles": smiles,
                "fingerprints": fingerprints,
                "fingerprint_type": fingerprint_type
            })
        except Exception as e:
            return json.dumps({"error": str(e)})