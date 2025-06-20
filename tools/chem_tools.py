from crewai.tools import BaseTool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, AllChem
import json
import requests

class PubChemLookupTool(BaseTool):
    name: str = "PubChem Compound Lookup"
    description: str = "Look up compound information from PubChem by SMILES, name, or CID. Returns chemical names, properties, and alternative identifiers."

    def _run(self, query: str, query_type: str = "smiles"):
        """
        query_type: 'smiles', 'name', or 'cid'
        """
        try:
            base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

            query_type=query_type.lower()
            if query_type == "smiles":
                url = f"{base_url}/compound/smiles/{query}/property/MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES,CanonicalSMILES/JSON"
            elif query_type == "name":
                url = f"{base_url}/compound/name/{query}/property/MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES,CanonicalSMILES/JSON"
            elif query_type == "cid":
                url = f"{base_url}/compound/cid/{query}/property/MolecularFormula,MolecularWeight,IUPACName,IsomericSMILES,CanonicalSMILES/JSON"
            else:
                return json.dumps({"error": "Invalid query_type. Use 'smiles', 'name', or 'cid'"})

            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                    prop = data['PropertyTable']['Properties'][0]
                    return json.dumps({
                        "found": True,
                        "cid": prop.get('CID'),
                        "molecular_formula": prop.get('MolecularFormula'),
                        "molecular_weight": prop.get('MolecularWeight'),
                        "iupac_name": prop.get('IUPACName'),
                        "canonical_smiles": prop.get('CanonicalSMILES'),
                        "isomeric_smiles": prop.get('IsomericSMILES')
                    })

            return json.dumps({"found": False, "error": "Compound not found in PubChem"})

        except Exception as e:
            return json.dumps({"found": False, "error": str(e)})


class MolecularFingerprintTool(BaseTool):
    name: str = "Molecular Fingerprint Generator"
    description: str = "Generate various molecular fingerprints (Morgan, MACCS, RDKit) for similarity analysis."

    def _run(self, smiles: str, fingerprint_type: str = "morgan"):
        """
        fingerprint_type: 'morgan', 'maccs', 'rdkit', 'topological'
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return json.dumps({"error": "Invalid SMILES"})

            fingerprints = {}

            if fingerprint_type in ["morgan", "all"]:
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints["morgan"] = morgan_fp.ToBitString()

            if fingerprint_type in ["maccs", "all"]:
                from rdkit.Chem import MACCSkeys
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                fingerprints["maccs"] = maccs_fp.ToBitString()

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

class Smilesvalidatortool(BaseTool):
    name: str = "SMILES Validator"
    description: str = "Validates if a SMILES string represents a valid molecule."

    def _run(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return json.dumps({"valid": False, "error": "Invalid SMILES syntax"})

            canonical_smiles = Chem.MolToSmiles(mol)
            return json.dumps({
                "valid": True,
                "canonical_smiles": canonical_smiles,
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds()
            })
        except Exception as e:
            return json.dumps({"valid": False, "error": str(e)})