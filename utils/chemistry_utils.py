from pathlib import Path

import requests
import pubchempy as pcp
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, inchi, Crippen, Lipinski, FilterCatalog
from rdkit.Chem import AllChem, DataStructs
from urllib.parse import quote
import selfies
import logging




# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger("chemistry_utils")

DATA_DIR = Path(__file__).parent.parent / "data"
FUNCTIONAL_GROUPS_PATH = DATA_DIR / "FunctionalGroups.txt"

def resolve_to_smiles(query: str, query_type: str = "auto") -> dict:
    """
    Convert chemical identifiers to canonical SMILES.
    Supports: auto, smiles, name, cid, cas, chembl, inchi, selfies.
    """
    if not query or not query.strip():
        return {"status": "error", "error": "Input is empty"}

    try:
        # Auto-detect input type if requested
        if query_type == "auto":
            if query.startswith("InChI="):
                query_type = "inchi"
            elif query.startswith("SELFIES=") or all(c in selfies.get_alphabet() for c in query):
                query_type = "selfies"
            elif any(c in query for c in "0123456789@=#$()[]+-\\/") or len(query) > 20:
                mol = Chem.MolFromSmiles(query)
                if mol:
                    return {
                        "status": "success",
                        "smiles": Chem.MolToSmiles(mol, canonical=True),
                        "source": "RDKit SMILES"
                    }
                # fallback to name if smiles parse fails
                query_type = "name"
            else:
                query_type = "name"

        if query_type == "smiles":
            mol = Chem.MolFromSmiles(query)
            if mol:
                return {
                    'status': 'success',
                    'smiles': Chem.MolToSmiles(mol, canonical=True),
                    'source': 'RDKit SMILES'
                }
            return {'status': 'error', 'error': 'Invalid SMILES input'}

        if query_type == "inchi":
            mol = Chem.MolFromInchi(query)
            if mol:
                return {
                    'status': 'success',
                    'smiles': Chem.MolToSmiles(mol, canonical=True),
                    'source': 'RDKit InChI'
                }
            return {'status': 'error', 'error': 'Invalid InChI input'}

        if query_type == "selfies":
            try:
                smi = selfies.decoder(query)
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    return {
                        'status': 'success',
                        'smiles': Chem.MolToSmiles(mol, canonical=True),
                        'source': 'SELFIES decode'
                    }
                return {'status': 'error', 'error': 'Invalid SELFIES input'}
            except Exception as e:
                return {'status': 'error', 'error': f'SELFIES decoding failed: {str(e)}'}

        # PubChem resolution for name, cid, cas
        if query_type in ["cid", "name", "cas"]:
            try:
                compounds = pcp.get_compounds(query, query_type)
                if compounds:
                    return {
                        'status': 'success',
                        'smiles': compounds[0].canonical_smiles,
                        'source': f'PubChem {query_type}'
                    }
            except Exception as e:
                logger.debug(f"PubChemPy failed for {query_type}: {str(e)}")

        # ChEMBL resolution
        if query_type == "chembl":
            try:
                molecule = new_client.molecule
                record = molecule.get(query)
                smiles = record.get("molecule_structures", {}).get("canonical_smiles")
                if smiles:
                    return {
                        'status': 'success',
                        'smiles': smiles,
                        'source': 'ChEMBL'
                    }
            except Exception:
                pass

        # Fallback PubChem REST API for name, cid
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
        endpoints = {
            "cid": f"{base_url}/cid/{query}/property/SMILES/JSON",
            "name": f"{base_url}/name/{query}/property/SMILES/JSON",
            "cas": f"{base_url}/name/{query}/property/SMILES/JSON",
        }

        if query_type in endpoints:
            try:
                response = requests.get(endpoints[query_type], timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    pubchem_smiles = data["PropertyTable"]["Properties"][0]["SMILES"]
                    mol = Chem.MolFromSmiles(pubchem_smiles)
                    if mol:
                        return {
                            'status': 'success',
                            'smiles': Chem.MolToSmiles(mol, canonical=True),
                            'source': 'PubChem REST'
                        }
            except Exception as e:
                logger.debug(f"PubChem REST API failed for {query_type}: {str(e)}")

        return {'status': 'error', 'error': f'Unable to resolve {query_type}: {query}'}

    except Exception as e:
        logger.error(f"Resolution error: {str(e)}")
        return {'status': 'error', 'error': f'Resolution error: {str(e)}'}


def validate_smiles(smiles: str) -> dict:
    """Validate SMILES and return canonical form"""
    try:
        if not smiles or not isinstance(smiles, str):
            return {"valid": False, "error": "Empty or invalid SMILES input"}

        smiles = smiles.strip()
        invalid_chars = set('$%^&;:"<>?\\|~`')
        if any(char in smiles for char in invalid_chars):
            return {"valid": False, "error": "Contains invalid characters"}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"valid": False, "error": "RDKit failed to parse SMILES"}

        if mol.GetNumAtoms() < 1:
            return {"valid": False, "error": "No atoms found in structure"}

        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

        return {
            "valid": True,
            "canonical_smiles": canonical_smiles,
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds()
        }

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {"valid": False, "error": f"Validation error: {str(e)}"}


def get_rdkit_properties(smiles: str) -> dict:
    """Get comprehensive properties from validated SMILES"""
    validation = validate_smiles(smiles)
    if not validation.get("valid", False):
        return validation

    try:
        mol = Chem.MolFromSmiles(validation["canonical_smiles"])
        return {
            **validation,
            "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
            "molecular_weight": Descriptors.MolWt(mol),
            "exact_molecular_weight": Descriptors.ExactMolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "h_bond_donors": rdMolDescriptors.CalcNumHBD(mol),
            "qed":Descriptors.qed(mol),
            "h_bond_acceptors": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "ring_count": rdMolDescriptors.CalcNumRings(mol)
        }
    except Exception as e:
        logger.error(f"Property calculation error: {str(e)}")
        return {"valid": False, "error": f"Property calculation error: {str(e)}"}


def resolve_and_validate(query: str, query_type: str) -> dict:
    """Combined resolution and validation"""
    resolution = resolve_to_smiles(query, query_type)
    if resolution["status"] != "success":
        return resolution
    return get_rdkit_properties(resolution["smiles"])


def find_similar_molecules(
        reference_smiles: str,
        similarity_threshold: float = 0.7,
        max_results: int = 5
) -> dict:
    """
    Find structurally similar molecules in PubChem using Tanimoto similarity
    """
    try:
        if not (0.0 <= similarity_threshold <= 1.0):
            return {"status": "error", "error": "similarity_threshold must be between 0 and 1"}
        if max_results <= 0:
            return {"status": "error", "error": "max_results must be greater than 0"}

        # Validate input SMILES
        validation = validate_smiles(reference_smiles)
        if not validation.get("valid"):
            return {"status": "error", "error": "Invalid reference SMILES"}

        canonical_smiles = validation["canonical_smiles"]
        ref_mol = Chem.MolFromSmiles(canonical_smiles)
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)

        # URL encode the SMILES for the API call
        encoded_smiles = quote(canonical_smiles)

        # Use GET request with proper URL encoding
        pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{encoded_smiles}/cids/JSON"
        pubchem_params = {
            "Threshold": int(similarity_threshold * 100),
            "MaxRecords": max_results * 5  # Request more to account for filtering
        }

        response = requests.get(pubchem_url, params=pubchem_params, timeout=30)
        if response.status_code != 200:
            return {"status": "error", "error": f"PubChem API error: {response.status_code}"}

        data = response.json()
        cids = data.get("IdentifierList", {}).get("CID", [])

        if not cids:
            return {
                "status": "success",
                "reference_smiles": canonical_smiles,
                "similarity_threshold": similarity_threshold,
                "results": []
            }

        results = []
        processed_smiles = set()  # Track processed SMILES to avoid duplicates
        processed_smiles.add(canonical_smiles)  # Add reference molecule to avoid returning it

        for cid in cids:
            try:
                compound_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"
                compound_resp = requests.get(compound_url, timeout=10)

                if compound_resp.status_code != 200:
                    continue

                compound_data = compound_resp.json()

                # Extract properties more safely
                props = compound_data["PC_Compounds"][0]["props"]

                # Find SMILES
                smiles = None
                for prop in props:
                    if prop["urn"]["label"] == "SMILES" and prop["urn"]["name"] == "Canonical":
                        smiles = prop["value"]["sval"]
                        break

                if not smiles:
                    # Fallback to any SMILES if canonical not found
                    for prop in props:
                        if prop["urn"]["label"] == "SMILES":
                            smiles = prop["value"]["sval"]
                            break

                if not smiles:
                    continue

                # Find IUPAC name
                name = "N/A"
                for prop in props:
                    if (prop["urn"]["label"] == "IUPAC Name" and
                            prop["urn"]["name"] == "Preferred"):
                        name = prop["value"]["sval"]
                        break

                # Process molecule
                hit_mol = Chem.MolFromSmiles(smiles)
                if not hit_mol:
                    continue

                canonical_hit_smiles = Chem.MolToSmiles(hit_mol)

                # Skip if we've already processed this molecule or it's the reference
                if canonical_hit_smiles in processed_smiles:
                    continue

                # Calculate similarity
                hit_fp = AllChem.GetMorganFingerprintAsBitVect(hit_mol, 2, nBits=2048)
                similarity = DataStructs.TanimotoSimilarity(ref_fp, hit_fp)

                # Add to results if it meets threshold
                if similarity >= similarity_threshold:
                    results.append({
                        "cid": cid,
                        "smiles": canonical_hit_smiles,
                        "name": name,
                        "similarity": round(similarity, 4)
                    })

                    processed_smiles.add(canonical_hit_smiles)

                    if len(results) >= max_results:
                        break

            except Exception as e:
                # Log the error if you have logging set up
                # print(f"Error processing CID {cid}: {str(e)}")
                continue

        return {
            "status": "success",
            "reference_smiles": canonical_smiles,
            "similarity_threshold": similarity_threshold,
            "results": sorted(results, key=lambda x: x["similarity"], reverse=True)
        }

    except Exception as e:
        return {"status": "error", "error": f"Similarity search failed: {str(e)}"}


# New helper functions for SELFIES conversion and molecular formula:
def smiles_to_selfies(smiles: str) -> dict:
    """Convert SMILES to SELFIES"""
    try:
        sf = selfies.encoder(smiles)
        return {"status": "success", "output": sf}
    except Exception as e:
        return {"status": "error", "error": f"SMILES to SELFIES conversion failed: {str(e)}"}


def selfies_to_smiles(sf: str) -> dict:
    """Convert SELFIES to SMILES"""
    try:
        smi = selfies.decoder(sf)
        return {"status": "success", "smiles": smi}
    except Exception as e:
        return {"status": "error", "error": f"SELFIES to SMILES conversion failed: {str(e)}"}


def calculate_molecular_formula(smiles: str) -> dict:
    """Calculate molecular formula from SMILES"""
    try:
        if not smiles or smiles.strip() == "":
            return {"status": "error", "error": "Empty or invalid SMILES input"}

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"status": "error", "error": "Invalid SMILES for formula calculation"}
        formula = rdMolDescriptors.CalcMolFormula(mol)
        return {"status": "success", "molecular_formula": formula}
    except Exception as e:
        return {"status": "error", "error": f"Formula calculation failed: {str(e)}"}


def get_iupac_name_from_smiles(smiles: str) -> dict:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {"status": "error", "error": f"PubChem REST API error: {resp.status_code}"}
        data = resp.json()
        iupac_name = data["PropertyTable"]["Properties"][0].get("IUPACName")
        if not iupac_name:
            return {"status": "error", "error": "No IUPAC name found"}
        return {"status": "success", "output": iupac_name}
    except Exception as e:
        return {"status": "error", "error": f"Request failed: {str(e)}"}


def get_common_name_from_smiles(smiles: str) -> dict:
    # Note: PubChem does not have a dedicated "common name" endpoint,
    # but we can try synonyms or fallback strategies.
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/synonyms/JSON"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {"status": "error", "error": f"PubChem REST API error: {resp.status_code}"}
        data = resp.json()
        synonyms = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
        if not synonyms:
            return {"status": "error", "error": "No synonyms found"}
        # Return the first synonym that looks like a common name (simple heuristic)
        common_name = synonyms[0]
        return {"status": "success", "output": common_name}
    except Exception as e:
        return {"status": "error", "error": f"Request failed: {str(e)}"}

def is_patented(smiles: str) -> dict:
    """
    Check if a molecule is associated with any patents via PubChem data.

    Args:
        smiles (str): Input SMILES string.

    Returns:
        dict: Patent check result including source and confidence.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"patented": False, "error": "Invalid SMILES"}

        inchikey = inchi.MolToInchiKey(mol)

        # Step 1: Get CID from InChIKey
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
        cid_response = requests.get(cid_url, timeout=10)

        if cid_response.status_code != 200:
            return {"patented": False, "error": "No CID found for molecule"}

        cids = cid_response.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return {"patented": False, "error": "Empty CID list"}

        cid = cids[0]

        # Step 2: Query PUG View Data for the CID
        view_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        view_response = requests.get(view_url, timeout=10)

        if view_response.status_code != 200:
            return {"patented": False, "error": "No PUG View data found"}

        data = view_response.json()
        sections = data.get("Record", {}).get("Section", [])

        # Look for "Patent" in any section title
        patent_sections = [
            section for section in sections
            if "Patent" in section.get("TOCHeading", "")
        ]

        return {
            "patented": bool(patent_sections),
            "source": "PubChem",
            "cid": cid,
            "inchikey": inchikey,
            "num_patent_sections_found": len(patent_sections)
        }

    except Exception as e:
        return {"patented": False, "error": f"Exception: {str(e)}"}

def check_drug_likeness(smiles: str) -> dict:
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"valid": False, "error": "Invalid SMILES"}

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)

        lipinski_pass = (mw < 500 and logp < 5 and h_donors <= 5 and h_acceptors <= 10)
        veber_pass = (rot_bonds <= 10 and tpsa <= 140)

        return {
            "molecular_weight": mw,
            "logP": logp,
            "H_donors": h_donors,
            "H_acceptors": h_acceptors,
            "rotatable_bonds": rot_bonds,
            "TPSA": tpsa,
            "lipinski_pass": lipinski_pass,
            "veber_pass": veber_pass
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def check_toxicity(smiles: str) -> dict:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"valid": False, "error": "Invalid SMILES"}

        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)

        catalog = FilterCatalog.FilterCatalog(params)
        entry = catalog.GetFirstMatch(mol)

        if entry is not None:
            return {
                "valid": True,
                "pains_alert": True,
                "alert_description": entry.GetDescription()
            }

        return {"valid": True, "pains_alert": False}

    except Exception as e:
        return {"valid": False, "error": str(e)}

def check_duplicate(smiles1: str, smiles2: str) -> dict:
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if not mol1 or not mol2:
            return {"valid": False, "error": "Invalid SMILES"}

        canon1 = Chem.MolToSmiles(mol1, canonical=True)
        canon2 = Chem.MolToSmiles(mol2, canonical=True)

        return {
            "canonical_smiles_1": canon1,
            "canonical_smiles_2": canon2,
            "is_duplicate": canon1 == canon2
        }
    except Exception as e:
        return {"error": str(e)}

def load_rdkit_functional_groups(filepath: str = FUNCTIONAL_GROUPS_PATH) -> dict:
    """
    Load functional groups SMARTS from RDKit FunctionalGroups.txt file.

    Args:
        filepath (str): Path to FunctionalGroups.txt

    Returns:
        dict: Mapping of functional group name to SMARTS pattern
    """
    fg_dict = {}
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Functional groups file not found: {filepath}")

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty and comment lines
            if not line or line.startswith("//"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                label = parts[0].strip()
                smarts = parts[1].strip()
                fg_dict[label] = smarts
    return fg_dict

