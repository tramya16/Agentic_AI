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

import numpy as np
import pickle
import gzip
from typing import List, Set, Optional
import time

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger("chemistry_utils")

DATA_DIR = Path(__file__).parent.parent / "data"
FUNCTIONAL_GROUPS_PATH = DATA_DIR / "FunctionalGroups.txt"

# Add hardcoded common molecules as fallback
COMMON_MOLECULES = {
    "albuterol": "CC(C)(C)NCC(O)c1ccc(O)cc1O",
    "salbutamol": "CC(C)(C)NCC(O)c1ccc(O)cc1O",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "penicillin": "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "tylenol": "CC(=O)Nc1ccc(O)cc1"
}


def resolve_to_smiles(query: str, query_type: str = "auto") -> dict:
    """
    Convert chemical identifiers to canonical SMILES.
    Supports: auto, smiles, name, cid, cas, chembl, inchi, selfies.
    """
    if not query or not query.strip():
        return {"status": "error", "error": "Input is empty"}

    try:
        query_lower = query.strip().lower()

        # Check hardcoded common molecules first for name queries
        if query_type in ["name", "auto"] and query_lower in COMMON_MOLECULES:
            smiles = COMMON_MOLECULES[query_lower]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return {
                    'status': 'success',
                    'smiles': Chem.MolToSmiles(mol, canonical=True),
                    'source': 'Hardcoded common molecule database'
                }

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
            except requests.exceptions.RequestException as e:
                logger.debug(f"PubChemPy API connection failed for {query_type}: {str(e)}")
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
            except requests.exceptions.RequestException as e:
                logger.debug(f"ChEMBL API connection failed: {str(e)}")
            except Exception as e:
                logger.debug(f"ChEMBL API failed: {str(e)}")

        # Fallback PubChem REST API for name, cid
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
        endpoints = {
            "cid": f"{base_url}/cid/{query}/property/CanonicalSMILES/JSON",
            "name": f"{base_url}/name/{quote(query)}/property/CanonicalSMILES/JSON",
            "cas": f"{base_url}/name/{quote(query)}/property/CanonicalSMILES/JSON",
        }

        if query_type in endpoints:
            try:
                response = requests.get(endpoints[query_type], timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    pubchem_smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
                    mol = Chem.MolFromSmiles(pubchem_smiles)
                    if mol:
                        return {
                            'status': 'success',
                            'smiles': Chem.MolToSmiles(mol, canonical=True),
                            'source': 'PubChem REST API'
                        }
                elif response.status_code == 404:
                    return {
                        'status': 'error',
                        'error': f'Molecule "{query}" not found in PubChem database'
                    }
                else:
                    return {
                        'status': 'error',
                        'error': f'PubChem API failed with status code {response.status_code}'
                    }
            except requests.exceptions.Timeout:
                return {
                    'status': 'error',
                    'error': 'PubChem API request timed out - service may be unavailable'
                }
            except requests.exceptions.ConnectionError:
                return {
                    'status': 'error',
                    'error': 'Cannot connect to PubChem API - check internet connection'
                }
            except requests.exceptions.RequestException as e:
                return {
                    'status': 'error',
                    'error': f'PubChem API request failed: {str(e)}'
                }
            except Exception as e:
                logger.debug(f"PubChem REST API failed for {query_type}: {str(e)}")
                return {
                    'status': 'error',
                    'error': f'PubChem API processing failed: {str(e)}'
                }

        return {
            'status': 'error',
            'error': f'Unable to resolve {query_type}: "{query}" - all API sources failed or unavailable'
        }

    except Exception as e:
        logger.error(f"Resolution error: {str(e)}")
        return {
            'status': 'error',
            'error': f'Unexpected error during resolution: {str(e)}'
        }


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

        try:
            response = requests.get(pubchem_url, params=pubchem_params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                cids = data.get("IdentifierList", {}).get("CID", [])
            elif response.status_code == 404:
                return {
                    "status": "success",
                    "reference_smiles": canonical_smiles,
                    "similarity_threshold": similarity_threshold,
                    "results": [],
                    "message": "No similar molecules found in PubChem"
                }
            else:
                return {
                    "status": "error",
                    "error": f"PubChem similarity API failed with status code {response.status_code}"
                }
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "error": "PubChem similarity API request timed out - service may be slow"
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "error": "Cannot connect to PubChem similarity API - check internet connection"
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"PubChem similarity API request failed: {str(e)}"
            }

        if not cids:
            return {
                "status": "success",
                "reference_smiles": canonical_smiles,
                "similarity_threshold": similarity_threshold,
                "results": [],
                "message": "No similar molecules found above threshold"
            }

        results = []
        processed_smiles = set()  # Track processed SMILES to avoid duplicates
        processed_smiles.add(canonical_smiles)  # Add reference molecule to avoid returning it
        api_failures = 0

        for cid in cids:
            try:
                compound_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"
                compound_resp = requests.get(compound_url, timeout=10)

                if compound_resp.status_code != 200:
                    api_failures += 1
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

            except requests.exceptions.RequestException:
                api_failures += 1
                continue
            except Exception as e:
                # Log the error if you have logging set up
                logger.debug(f"Error processing CID {cid}: {str(e)}")
                continue

        result = {
            "status": "success",
            "reference_smiles": canonical_smiles,
            "similarity_threshold": similarity_threshold,
            "results": sorted(results, key=lambda x: x["similarity"], reverse=True)
        }

        if api_failures > 0:
            result["warnings"] = f"{api_failures} compound API calls failed during similarity search"

        return result

    except Exception as e:
        return {"status": "error", "error": f"Similarity search failed: {str(e)}"}


def get_iupac_name_from_smiles(smiles: str) -> dict:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/property/IUPACName/JSON"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            iupac_name = data["PropertyTable"]["Properties"][0].get("IUPACName")
            if not iupac_name:
                return {"status": "error", "error": "No IUPAC name found in PubChem"}
            return {"status": "success", "output": iupac_name}
        elif resp.status_code == 404:
            return {"status": "error", "error": "Molecule not found in PubChem database"}
        else:
            return {"status": "error", "error": f"PubChem API failed with status code {resp.status_code}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "PubChem API request timed out"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "Cannot connect to PubChem API - check internet connection"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": f"PubChem API request failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "error": f"Request processing failed: {str(e)}"}


def get_common_name_from_smiles(smiles: str) -> dict:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/synonyms/JSON"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            synonyms = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
            if not synonyms:
                return {"status": "error", "error": "No synonyms found in PubChem"}
            # Return the first synonym that looks like a common name (simple heuristic)
            common_name = synonyms[0]
            return {"status": "success", "output": common_name}
        elif resp.status_code == 404:
            return {"status": "error", "error": "Molecule not found in PubChem database"}
        else:
            return {"status": "error", "error": f"PubChem API failed with status code {resp.status_code}"}
    except requests.exceptions.Timeout:
        return {"status": "error", "error": "PubChem API request timed out"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "error": "Cannot connect to PubChem API - check internet connection"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": f"PubChem API request failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "error": f"Request processing failed: {str(e)}"}


# Keep all other functions the same...
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
            "qed": Descriptors.qed(mol),
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


# Keep all other functions unchanged...
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


def is_patented(smiles: str) -> dict:
    """
    Check if a molecule is associated with any patents via PubChem data.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"patented": False, "error": "Invalid SMILES"}

        inchikey = inchi.MolToInchiKey(mol)

        # Step 1: Get CID from InChIKey
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/JSON"
        try:
            cid_response = requests.get(cid_url, timeout=10)
            if cid_response.status_code != 200:
                return {"patented": False, "error": f"PubChem API failed with status {cid_response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"patented": False, "error": f"PubChem API connection failed: {str(e)}"}

        cids = cid_response.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return {"patented": False, "error": "No CID found for molecule in PubChem"}

        cid = cids[0]

        # Step 2: Query PUG View Data for the CID
        view_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        try:
            view_response = requests.get(view_url, timeout=10)
            if view_response.status_code != 200:
                return {"patented": False,
                        "error": f"PubChem PUG View API failed with status {view_response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"patented": False, "error": f"PubChem PUG View API connection failed: {str(e)}"}

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
        return {"patented": False, "error": f"Patent check failed: {str(e)}"}


def check_drug_likeness(smiles: str) -> dict:
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


# Add these new functions to your existing chemistry_utils.py

def download_zinc250k_smiles(cache_dir: Path = None) -> Set[str]:
    """
    Download and cache ZINC250k dataset for novelty checking.

    Returns:
        Set of canonical SMILES from ZINC250k

    Dataset Reference:
        This function downloads the curated ZINC250k subset used in the Chemical VAE paper:
        Gómez-Bombarelli, R., Wei, J.N., Duvenaud, D., et al.
        "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules"
        ACS Central Science, 2018. DOI: 10.1021/acscentsci.7b00572

        Dataset Source:
        https://github.com/aspuru-guzik-group/chemical_vae/blob/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv

        This subset has been cleaned for duplicates and invalid molecules and is widely used for
        benchmarking generative models (e.g., in MOSES, GuacaMol, GraphVAE, JT-VAE).

    Notes:
        - The dataset contains ~250,000 SMILES entries.
        - Canonical SMILES are computed using RDKit.
        - Results are cached in a gzip-compressed pickle file.
        - Invalid or unparsable SMILES are automatically skipped.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data" / "reference_datasets"

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "zinc250k_smiles.pkl.gz"

    if cache_file.exists():
        print(f"Loading ZINC250k from cache: {cache_file}")
        with gzip.open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Downloading ZINC250k dataset...")
    # Using the GuacaMol reference URL
    # Citation: https://github.com/BenevolentAI/guacamol/blob/master/guacamol/standard_benchmarks.py
    zinc_url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"

    try:
        response = requests.get(zinc_url, timeout=300)  # 5 min timeout
        response.raise_for_status()

        zinc_smiles = set()
        lines = response.text.strip().split('\n')

        for i, line in enumerate(lines[1:], 1):  # Skip header
            if i % 10000 == 0:
                print(f"Processed {i} ZINC molecules...")

            parts = line.split(',')
            if len(parts) > 0:
                smiles = parts[0].strip().strip('"')
                if smiles:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                            zinc_smiles.add(canonical_smiles)
                    except:
                        continue

        print(f"Successfully processed {len(zinc_smiles)} ZINC molecules")

        # Cache the results
        with gzip.open(cache_file, 'wb') as f:
            pickle.dump(zinc_smiles, f)
        print(f"Cached ZINC250k to: {cache_file}")

        return zinc_smiles

    except Exception as e:
        print(f"Failed to download ZINC250k: {e}")
        return set()


def enhanced_find_similar_molecules(
        reference_smiles: str,
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        use_chembl: bool = True,
        use_pubchem: bool = True
) -> dict:
    """
    Enhanced similarity search using multiple databases and improved strategies.

    Improvements over original:
    1. Lower initial threshold for PubChem API
    2. Multiple database sources
    3. Better error handling
    4. Substructure search fallback
    """
    try:
        # Validate input
        validation = validate_smiles(reference_smiles)
        if not validation.get("valid"):
            return {"status": "error", "error": "Invalid reference SMILES"}

        canonical_smiles = validation["canonical_smiles"]
        ref_mol = Chem.MolFromSmiles(canonical_smiles)
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)

        all_results = []

        # Strategy 1: PubChem similarity with lower threshold
        if use_pubchem:
            pubchem_results = _pubchem_similarity_search(
                canonical_smiles,
                max(0.5, similarity_threshold - 0.2),  # Lower threshold for API
                max_results * 3
            )
            if pubchem_results.get("results"):
                all_results.extend(pubchem_results["results"])

        # Strategy 2: ChEMBL similarity search
        if use_chembl:
            chembl_results = _chembl_similarity_search(
                canonical_smiles,
                similarity_threshold,
                max_results
            )
            if chembl_results.get("results"):
                all_results.extend(chembl_results["results"])

        # Strategy 3: Substructure search fallback if no results
        if not all_results:
            substructure_results = _pubchem_substructure_search(
                canonical_smiles,
                max_results
            )
            if substructure_results.get("results"):
                # Calculate similarities for substructure matches
                for result in substructure_results["results"]:
                    try:
                        hit_mol = Chem.MolFromSmiles(result["smiles"])
                        if hit_mol:
                            hit_fp = AllChem.GetMorganFingerprintAsBitVect(hit_mol, 2, nBits=2048)
                            similarity = DataStructs.TanimotoSimilarity(ref_fp, hit_fp)
                            result["similarity"] = similarity
                            if similarity >= similarity_threshold:
                                all_results.append(result)
                    except:
                        continue

        # Remove duplicates and filter by threshold
        seen_smiles = set()
        filtered_results = []

        for result in all_results:
            smiles = result.get("smiles")
            similarity = result.get("similarity", 0)

            if (smiles and smiles not in seen_smiles and
                    similarity >= similarity_threshold and
                    smiles != canonical_smiles):  # Exclude reference molecule

                seen_smiles.add(smiles)
                filtered_results.append(result)

        # Sort by similarity and limit results
        filtered_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        final_results = filtered_results[:max_results]

        return {
            "status": "success",
            "reference_smiles": canonical_smiles,
            "similarity_threshold": similarity_threshold,
            "results": final_results,
            "total_found": len(filtered_results),
            "sources_used": ["PubChem" if use_pubchem else None,
                             "ChEMBL" if use_chembl else None],
            "search_strategies": ["similarity", "substructure"] if not final_results else ["similarity"]
        }

    except Exception as e:
        return {"status": "error", "error": f"Enhanced similarity search failed: {str(e)}"}


def _pubchem_similarity_search(smiles: str, threshold: float, max_results: int) -> dict:
    """PubChem similarity search with improved parameters"""
    try:
        encoded_smiles = quote(smiles)

        # Use lower threshold for API call to get more candidates
        api_threshold = max(50, int(threshold * 100))  # Minimum 50% for API

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{encoded_smiles}/cids/JSON"
        params = {
            "Threshold": api_threshold,
            "MaxRecords": min(max_results * 2, 200)  # Get more candidates
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])

            results = []
            ref_mol = Chem.MolFromSmiles(smiles)
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)

            for cid in cids[:max_results]:
                try:
                    # Get compound data
                    compound_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,IUPACName/JSON"
                    comp_resp = requests.get(compound_url, timeout=10)

                    if comp_resp.status_code == 200:
                        comp_data = comp_resp.json()
                        props = comp_data["PropertyTable"]["Properties"][0]

                        hit_smiles = props.get("CanonicalSMILES")
                        name = props.get("IUPACName", "N/A")

                        if hit_smiles:
                            hit_mol = Chem.MolFromSmiles(hit_smiles)
                            if hit_mol:
                                hit_fp = AllChem.GetMorganFingerprintAsBitVect(hit_mol, 2, nBits=2048)
                                similarity = DataStructs.TanimotoSimilarity(ref_fp, hit_fp)

                                results.append({
                                    "cid": cid,
                                    "smiles": hit_smiles,
                                    "name": name,
                                    "similarity": round(similarity, 4),
                                    "source": "PubChem"
                                })

                    time.sleep(0.1)  # Rate limiting

                except Exception:
                    continue

            return {"status": "success", "results": results}

        return {"status": "error", "results": []}

    except Exception as e:
        return {"status": "error", "error": str(e), "results": []}


def _chembl_similarity_search(smiles: str, threshold: float, max_results: int) -> dict:
    """ChEMBL similarity search"""
    try:
        # ChEMBL similarity API endpoint
        # Citation: ChEMBL web services - https://chembl.gitbook.io/chembl-interface-documentation/web-services
        url = "https://www.ebi.ac.uk/chembl/api/data/similarity"

        params = {
            "smiles": smiles,
            "similarity": int(threshold * 100),
            "limit": max_results
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            molecules = data.get("molecules", [])

            results = []
            for mol in molecules:
                mol_smiles = mol.get("molecule_structures", {}).get("canonical_smiles")
                if mol_smiles:
                    results.append({
                        "chembl_id": mol.get("molecule_chembl_id"),
                        "smiles": mol_smiles,
                        "name": mol.get("pref_name", "N/A"),
                        "similarity": mol.get("similarity", 0) / 100.0,
                        "source": "ChEMBL"
                    })

            return {"status": "success", "results": results}

        return {"status": "error", "results": []}

    except Exception as e:
        return {"status": "error", "error": str(e), "results": []}


def _pubchem_substructure_search(smiles: str, max_results: int) -> dict:
    """PubChem substructure search as fallback"""
    try:
        encoded_smiles = quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/substructure/smiles/{encoded_smiles}/cids/JSON"

        params = {"MaxRecords": max_results}

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])

            results = []
            for cid in cids[:max_results]:
                try:
                    compound_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
                    comp_resp = requests.get(compound_url, timeout=10)

                    if comp_resp.status_code == 200:
                        comp_data = comp_resp.json()
                        hit_smiles = comp_data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]

                        results.append({
                            "cid": cid,
                            "smiles": hit_smiles,
                            "name": "N/A",
                            "similarity": 0.0,  # Will be calculated later
                            "source": "PubChem_substructure"
                        })

                    time.sleep(0.1)

                except Exception:
                    continue

            return {"status": "success", "results": results}

        return {"status": "error", "results": []}

    except Exception as e:
        return {"status": "error", "error": str(e), "results": []}


def calculate_novelty_against_zinc(generated_smiles: List[str], zinc_smiles: Set[str] = None) -> dict:
    """
    Calculate novelty of generated molecules against ZINC250k dataset.

    Args:
        generated_smiles: List of generated SMILES
        zinc_smiles: Set of ZINC SMILES (will download if None)

    Returns:
        Dictionary with novelty metrics
    """
    try:
        if zinc_smiles is None:
            print("Loading ZINC250k dataset for novelty calculation...")
            zinc_smiles = download_zinc250k_smiles()
            print(f"✅ Loaded {len(zinc_smiles)} unique canonical SMILES from ZINC250k.")

        if not zinc_smiles:
            return {"status": "error", "error": "Failed to load ZINC250k dataset"}

        # Canonicalize generated SMILES
        canonical_generated = set()
        invalid_smiles = []

        for smiles in generated_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canonical = Chem.MolToSmiles(mol, canonical=True)
                    canonical_generated.add(canonical)
                else:
                    invalid_smiles.append(smiles)
            except:
                invalid_smiles.append(smiles)

        # Calculate novelty
        novel_molecules = canonical_generated - zinc_smiles
        known_molecules = canonical_generated & zinc_smiles

        novelty_rate = len(novel_molecules) / len(canonical_generated) if canonical_generated else 0

        return {
            "status": "success",
            "total_generated": len(generated_smiles),
            "valid_generated": len(canonical_generated),
            "invalid_generated": len(invalid_smiles),
            "novel_molecules": list(novel_molecules),
            "known_molecules": list(known_molecules),
            "novel_count": len(novel_molecules),
            "known_count": len(known_molecules),
            "novelty_rate": novelty_rate,
            "zinc_reference_size": len(zinc_smiles)
        }

    except Exception as e:
        return {"status": "error", "error": f"Novelty calculation failed: {str(e)}"}