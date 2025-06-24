from pydantic import BaseModel,Field
from crewai.tools import BaseTool
from typing import Literal, Optional, Type
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, AllChem, MolStandardize, Draw
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import json
import requests


class PubChemLookupInput(BaseModel):
    """Input parameters for PubChem lookup tool"""
    query: str
    query_type: Literal["smiles", "name", "cid"] = "smiles"


class PubChemLookupTool(BaseTool):
    name: str = "pubchem_compound_lookup"
    description: str = (
        "Retrieves chemical compound data from PubChem. "
        "Accepts SMILES strings, compound names, or PubChem CIDs. "
        "Returns molecular properties, identifiers, and names. "
        "Use query_type='smiles' for chemical structures, "
        "'name' for compound names, or 'cid' for PubChem IDs."
    )
    args_schema: Type[BaseModel] = PubChemLookupInput

    def _run(
            self,
            query: str,
            query_type: Literal["smiles", "name", "cid"] = "smiles"
    ) -> str:
        """
        Look up compound information in PubChem

        Args:
            query: Chemical identifier (SMILES, name, or CID)
            query_type: Type of identifier being provided

        Returns:
            JSON string with structured results or error message
        """
        query = query.strip()
        if not query:
            return json.dumps({"status": "error", "error": "Query is empty or only whitespace"})

        valid_types = {"name", "cid", "smiles"}
        if query_type not in valid_types:
            return json.dumps({
                "status": "error",
                "error": "Invalid query_type. Use 'smiles', 'name', or 'cid'"
            })

        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
        properties = "MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,IsomericSMILES"
        url = f"{base_url}/{query_type}/{query}/property/{properties}/JSON"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            props = data.get("PropertyTable", {})
            property_list = props.get("Properties", [])

            if not property_list:
                return json.dumps({
                    "status": "error",
                    "error": "Compound not found in PubChem",
                    "found": False,
                    "cid": query if query_type == "cid" else None
                })

            properties = property_list[0]
            result = {
                "status": "success",
                "found": True,
                "cid": properties.get("CID"),
                "molecular_formula": properties.get("MolecularFormula", "N/A"),
                "molecular_weight": properties.get("MolecularWeight", "N/A"),
                "iupac_name": properties.get("IUPACName", "N/A"),
                "canonical_smiles": properties.get("CanonicalSMILES", "N/A"),
                "isomeric_smiles": properties.get("IsomericSMILES", "N/A")
            }
            return json.dumps(result)

        except requests.exceptions.HTTPError as http_err:
            error_message = "Compound not found in PubChem (404)" if http_err.response.status_code == 404 else f"HTTP error: {http_err}"
            return json.dumps({
                "status": "error",
                "error": error_message,
                "found": False,
                "cid": query if query_type == "cid" else None
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
                "found": False,
                "cid": query if query_type == "cid" else None
            })


class NameConversionInput(BaseModel):
    """Input parameters for chemical name conversion"""
    input_value: str = Field(..., description="Chemical name or SMILES string to convert")
    input_type: Literal["auto", "name", "smiles"] = Field(
        default="auto",
        description="Input type: 'auto' to detect automatically, 'name' for chemical names, 'smiles' for SMILES strings"
    )
    output_type: Literal["iupac_name", "common_name", "smiles"] = Field(
        default="iupac_name",
        description="Desired output format: 'iupac_name', 'common_name', or 'smiles'"
    )


class ChemicalNameConverterTool(BaseTool):
    name: str = "chemical_name_converter"
    description: str = (
        "Converts between chemical names and SMILES representations. "
        "Input can be IUPAC/common names or SMILES strings. "
        "Output can be IUPAC name, common name, or canonical SMILES."
    )
    args_schema: Type[BaseModel] = NameConversionInput

    def _run(
            self,
            input_value: str,
            input_type: str = "auto",
            output_type: str = "iupac_name"
    ) -> dict:
        """
        Converts chemical identifiers between names and SMILES formats

        Args:
            input_value: Chemical identifier to convert
            input_type: Input type ('auto', 'name', or 'smiles')
            output_type: Desired output format ('iupac_name', 'common_name', 'smiles')

        Returns:
            dict: Structured conversion result or error message
        """
        # Auto-detect input type if needed
        if input_type == "auto":
            input_type = self._detect_input_type(input_value)

        try:
            # Name to SMILES conversion
            if input_type == "name" and output_type == "smiles":
                return self._name_to_smiles(input_value)

            # SMILES to Name conversion
            elif input_type == "smiles" and output_type in ["iupac_name", "common_name"]:
                return self._smiles_to_name(input_value, output_type)

            # Unsupported conversion
            else:
                return {
                    "success": False,
                    "error": f"Unsupported conversion: {input_type} to {output_type}",
                    "supported_conversions": [
                        "name → smiles",
                        "smiles → iupac_name",
                        "smiles → common_name"
                    ]
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Conversion error: {str(e)}",
                "input_value": input_value,
                "input_type": input_type,
                "output_type": output_type
            }

    def _detect_input_type(self, value: str) -> str:
        """Automatically detect chemical identifier type"""
        # Simple heuristics for detection
        if " " in value or value.replace("-", "").replace("(", "").replace(")", "").isalpha():
            return "name"
        return "smiles"

    def _name_to_smiles(self, name: str) -> dict:
        """Convert chemical name to canonical SMILES"""
        from tools.chem_tools import PubChemLookupTool  # Local import to avoid circular dependency

        result = PubChemLookupTool()._run(name, "name")
        pubchem_data = json.loads(result)

        if not pubchem_data.get("found"):
            return {
                "success": False,
                "error": "Compound not found in PubChem",
                "input_name": name
            }

        return {
            "success": True,
            "input": name,
            "output_type": "smiles",
            "smiles": pubchem_data.get("canonical_smiles"),
            "source": "PubChem"
        }

    def _smiles_to_name(self, smiles: str, name_type: str) -> dict:
        """Convert SMILES to IUPAC or common name"""
        from tools.chem_tools import SmilesValidatorTool, PubChemLookupTool

        # Validate SMILES first
        validation = SmilesValidatorTool()._run(smiles)
        valid_data = json.loads(validation)

        if not valid_data.get("valid"):
            return {
                "success": False,
                "error": "Invalid SMILES string",
                "input_smiles": smiles,
                "validation_error": valid_data.get("error", "")
            }

        # Lookup in PubChem
        result = PubChemLookupTool()._run(smiles, "smiles")
        pubchem_data = json.loads(result)

        if not pubchem_data.get("found"):
            return {
                "success": False,
                "error": "Compound not found in PubChem",
                "input_smiles": smiles
            }

        # Return requested name type
        if name_type == "iupac_name":
            name_value = pubchem_data.get("iupac_name")
        else:  # common_name
            name_value = pubchem_data.get("iupac_name")  # PubChem doesn't return common names directly

        return {
            "success": True,
            "input": smiles,
            "output_type": name_type,
            "name": name_value,
            "source": "PubChem"
        }



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


class SmilesValidatorTool(BaseTool):
    name: str = "SMILES Validator"
    description: str = "Validates if a SMILES string represents a valid molecule."

    def _run(self, smiles: str):
        try:
            # Enhanced validation
            if not smiles or not isinstance(smiles, str):
                return json.dumps({"valid": False, "error": "Empty or invalid SMILES input"})

            # Check for obviously invalid characters
            if any(char in smiles for char in ['@#$%^&*()', '  ']):  # Multiple spaces
                return json.dumps({"valid": False, "error": "Contains invalid characters"})

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


class MolecularPropertyCalculator(BaseTool):
    name: str = "Molecular Property Calculator"
    description: str = "Calculate key molecular properties for drug discovery"

    def _run(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return json.dumps({"error": "Invalid SMILES"})

            return json.dumps({
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "h_bond_donors": Descriptors.NumHDonors(mol),
                "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "tpsa": round(Descriptors.TPSA(mol), 2),
                "qed_score": round(QED.qed(mol), 3),
                "formal_charge": Chem.GetFormalCharge(mol),
                "heavy_atoms": Descriptors.HeavyAtomCount(mol)
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


class FunctionalGroupAnalyzer(BaseTool):
    name: str = "Functional Group Analyzer"
    description: str = "Identify functional groups in a molecule"

    def _run(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return json.dumps({"error": "Invalid SMILES"})

            # Common functional groups
            groups = {
                "carboxylic_acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
                "ester": Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]"),
                "amide": Chem.MolFromSmarts("[CX3](=O)[NX3]"),
                "amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),
                "alcohol": Chem.MolFromSmarts("[OX2H][#6;!$(C=O)]"),
                "ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
                "aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
                "ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
                "nitrile": Chem.MolFromSmarts("C#N"),
                "halogen": Chem.MolFromSmarts("[F,Cl,Br,I]")
            }

            results = {}
            for name, pattern in groups.items():
                if pattern:  # Make sure pattern is valid
                    results[name] = len(mol.GetSubstructMatches(pattern))
                else:
                    results[name] = 0

            return json.dumps(results)
        except Exception as e:
            return json.dumps({"error": str(e)})


class DrugLikenessEvaluator(BaseTool):
    name: str = "Drug Likeness Evaluator"
    description: str = "Evaluate drug-likeness using multiple criteria"

    def _run(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return json.dumps({"error": "Invalid SMILES"})

            # Lipinski's Rule of Five
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            lipinski_pass = all([
                mw <= 500,
                logp <= 5,
                hbd <= 5,
                hba <= 10
            ])

            # Ghose filter - FIXED: Use mol.GetNumAtoms() instead of Descriptors.NumAtoms()
            num_atoms = mol.GetNumAtoms()  # CORRECTED LINE
            ghose_pass = all([
                160 <= mw <= 480,
                -0.4 <= logp <= 5.6,
                20 <= num_atoms <= 70
            ])

            # Veber rules
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            tpsa = Descriptors.TPSA(mol)
            veber_pass = (rotatable_bonds <= 10 and tpsa <= 140)

            # PAINS filter with improved error handling
            pains_hit = False
            try:
                params = FilterCatalogParams()
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
                catalog = FilterCatalog(params)
                pains_hit = catalog.HasMatch(mol)
            except Exception:
                # If PAINS filter fails, continue without it
                pains_hit = False  # Default to no PAINS alert if filter unavailable

            return json.dumps({
                "lipinski_ro5": lipinski_pass,
                "ghose_filter": ghose_pass,
                "veber_rules": veber_pass,
                "pains_alert": pains_hit,
                "qed_score": round(QED.qed(mol), 3),
                "properties": {
                    "molecular_weight": round(mw, 2),
                    "logp": round(logp, 2),
                    "h_bond_donors": hbd,
                    "h_bond_acceptors": hba,
                    "rotatable_bonds": rotatable_bonds,
                    "tpsa": round(tpsa, 2),
                    "num_atoms": num_atoms  # CORRECTED
                }
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


class MolecularSimilarityTool(BaseTool):
    name: str = "Molecular Similarity Calculator"
    description: str = "Calculate similarity between two molecules using fingerprints"

    def _run(self, smiles1: str, smiles2: str, fingerprint_type: str = "morgan"):
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if not mol1 or not mol2:
                return json.dumps({"error": "Invalid SMILES input"})

            # Generate fingerprints with improved error handling
            try:
                if fingerprint_type == "morgan":
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
                elif fingerprint_type == "maccs":
                    try:
                        from rdkit.Chem import MACCSkeys
                        fp1 = MACCSkeys.GenMACCSKeys(mol1)
                        fp2 = MACCSkeys.GenMACCSKeys(mol2)
                    except ImportError:
                        # Fallback to Morgan if MACCS not available
                        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
                        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
                        fingerprint_type = "morgan_fallback"
                else:  # Default to RDKit
                    fp1 = Chem.RDKFingerprint(mol1)
                    fp2 = Chem.RDKFingerprint(mol2)

                # Calculate Tanimoto similarity
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

                return json.dumps({
                    "similarity": round(similarity, 3),
                    "fingerprint_type": fingerprint_type,
                    "smiles1": smiles1,
                    "smiles2": smiles2
                })
            except Exception as fp_error:
                return json.dumps({"error": f"Fingerprint generation failed: {str(fp_error)}"})

        except Exception as e:
            return json.dumps({"error": str(e)})


class SimilarMoleculeFinderTool(BaseTool):
    name: str = "Similar Molecule Finder"
    description: str = "Find structurally similar molecules in PubChem, calculate their Tanimoto similarity to the reference, and return results with scores."

    def _run(self, smiles: str, similarity_threshold: float = 0.7, max_results: int = 5):
        try:
            # 1. Validate SMILES and get canonical form
            validation_tool = SmilesValidatorTool()  # Instantiate the tool
            validation_result = json.loads(validation_tool._run(smiles))
            if not validation_result.get("valid"):
                return json.dumps({"error": "Invalid reference SMILES provided."})

            reference_canonical_smiles = validation_result.get("canonical_smiles")

            # 2. Use PubChem's fastsimilarity_2d to get a list of CIDs
            pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{reference_canonical_smiles}/cids/JSON"
            pubchem_params = {
                "Threshold": int(similarity_threshold * 100),
                "MaxRecords": max_results * 2
                # Fetch more than needed initially to allow for sorting by calculated similarity
            }

            try:
                response = requests.post(pubchem_url, params=pubchem_params, timeout=30)
                response.raise_for_status()  # Raise an exception for bad status codes
                data = response.json()
                cids = data.get("IdentifierList", {}).get("CID", [])
            except requests.exceptions.Timeout:
                return json.dumps({"error": "PubChem API request timed out."})
            except requests.exceptions.RequestException as e:
                return json.dumps({"error": f"PubChem API request failed: {str(e)}"})
            except json.JSONDecodeError:
                return json.dumps({"error": "Failed to decode JSON from PubChem API response."})

            # 3. Get details for each CID and calculate similarity
            found_molecules_with_scores = []
            pubchem_lookup_tool = PubChemLookupTool()  # Instantiate the tool
            molecular_similarity_tool = MolecularSimilarityTool()  # Instantiate the tool

            for cid in cids:
                try:
                    # Get SMILES for the found CID
                    pubchem_data = json.loads(pubchem_lookup_tool._run(str(cid), "cid"))
                    if pubchem_data.get("found") and pubchem_data.get("canonical_smiles"):
                        hit_smiles = pubchem_data.get("canonical_smiles")

                        # Calculate Tanimoto similarity using your MolecularSimilarityTool
                        similarity_result = json.loads(
                            molecular_similarity_tool._run(reference_canonical_smiles, hit_smiles, "morgan")
                        )

                        if "similarity" in similarity_result:
                            found_molecules_with_scores.append({
                                "cid": cid,
                                "smiles": hit_smiles,
                                "iupac_name": pubchem_data.get("iupac_name", "N/A"),
                                "source": "PubChem",
                                "similarity": similarity_result["similarity"]
                            })
                except Exception as e:
                    # Log or print errors for individual CIDs but don't stop the whole process
                    print(f"Warning: Could not process CID {cid} for similarity calculation: {e}")
                    continue

            # 4. Sort results by similarity (descending) and take top N
            final_results = sorted(
                found_molecules_with_scores,
                key=lambda x: x.get("similarity", 0),  # Use 0 as default if similarity somehow missing
                reverse=True
            )[:max_results]

            return json.dumps({
                "reference_smiles": reference_canonical_smiles,
                "similarity_threshold": similarity_threshold,
                "results": final_results
            })

        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred in SimilarMoleculeFinderTool: {str(e)}"})


# =============================================================================
# STRUCTURE AGENT TOOLS
# =============================================================================

class ScaffoldAnalysisTool(BaseTool):
    name: str = "Scaffold Analysis Tool"
    description: str = "Analyze molecular scaffolds using Murcko scaffolds and identify core structures."

    def _run(self, smiles: str):
        try:
            try:
                from rdkit.Chem.Scaffolds import MurckoScaffold
            except ImportError:
                return json.dumps({"error": "MurckoScaffold module not available"})

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return json.dumps({"error": "Invalid SMILES"})

            # Get Murcko scaffold
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else None

                # Get framework (scaffold without side chains)
                framework = MurckoScaffold.MakeScaffoldGeneric(scaffold) if scaffold else None
                framework_smiles = Chem.MolToSmiles(framework) if framework else None
            except Exception:
                scaffold_smiles = None
                framework_smiles = None

            # Ring analysis
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            ring_sizes = [len(ring) for ring in ring_info.AtomRings()]

            return json.dumps({
                "smiles": smiles,
                "scaffold_smiles": scaffold_smiles,
                "framework_smiles": framework_smiles,
                "num_rings": num_rings,
                "ring_sizes": ring_sizes,
                "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "num_aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol)
            })

        except Exception as e:
            return json.dumps({"error": str(e)})


class SolubilityPredictor(BaseTool):
    name: str = "Solubility Predictor"
    description: str = "Predict aqueous solubility using various empirical models and molecular descriptors."

    def _run(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return json.dumps({"error": "Invalid SMILES"})

            # Calculate descriptors relevant to solubility
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

            # Simple solubility predictions (empirical models)
            # Ali et al. model: logS = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*aromatic_rings
            ali_logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * aromatic_rings

            # Huuskonen model approximation
            huuskonen_logs = 0.21 - 0.48 * logp - 0.003 * mw

            # Average prediction
            avg_logs = (ali_logs + huuskonen_logs) / 2

            # Solubility categories
            if avg_logs > -1:
                solubility_class = "Highly soluble"
            elif avg_logs > -3:
                solubility_class = "Moderately soluble"
            elif avg_logs > -5:
                solubility_class = "Poorly soluble"
            else:
                solubility_class = "Very poorly soluble"

            return json.dumps({
                "smiles": smiles,
                "predicted_logS": round(avg_logs, 3),
                "solubility_class": solubility_class,
                "ali_model_logS": round(ali_logs, 3),
                "huuskonen_model_logS": round(huuskonen_logs, 3),
                "molecular_weight": round(mw, 2),
                "logp": round(logp, 2),
                "tpsa": round(tpsa, 2),
                "hbd": hbd,
                "hba": hba
            })

        except Exception as e:
            return json.dumps({"error": str(e)})


# =============================================================================
# FRAGMENT AGENT TOOLS
# =============================================================================

class FunctionalGroupManager(BaseTool):
    name: str = "Functional Group Manager"
    description: str = "Advanced functional group analysis and modification suggestions using SMARTS patterns."

    def _run(self, smiles: str, operation: str = "analyze"):
        """
        operation: 'analyze', 'suggest_additions', 'suggest_removals'
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return json.dumps({"error": "Invalid SMILES"})

            # Extended functional group library
            functional_groups = {
                # Basic groups
                "carboxylic_acid": "[CX3](=O)[OX2H1]",
                "ester": "[CX3](=O)[OX2H0]",
                "amide": "[CX3](=[OX1])[NX3]",
                "primary_amine": "[NX3;H2;!$(NC=O)]",
                "secondary_amine": "[NX3;H1;!$(NC=O)]",
                "tertiary_amine": "[NX3;H0;!$(NC=O)]",
                "primary_alcohol": "[CX4][OX2H]",
                "secondary_alcohol": "[CX4H2][OX2H]",
                "tertiary_alcohol": "[CX4H0][OX2H]",
                "phenol": "[OX2H][cX3]:[c]",
                "ketone": "[CX3]=[OX1]",
                "aldehyde": "[CX3H1](=O)",
                "ether": "[OD2]([#6])[#6]",

                # Aromatic systems
                "benzene": "c1ccccc1",
                "pyridine": "c1ccncc1",
                "pyrimidine": "c1cncnc1",
                "imidazole": "c1c[nH]cn1",
                "furan": "c1ccoc1",
                "thiophene": "c1ccsc1",
                "indole": "c1ccc2[nH]ccc2c1",

                # Other important groups
                "nitrile": "[CX2]#N",
                "nitro": "[NX3+](=O)[O-]",
                "sulfoxide": "[SX3](=O)",
                "sulfone": "[SX4](=O)(=O)",
                "halogen": "[F,Cl,Br,I]",
                "trifluoromethyl": "C(F)(F)F"
            }

            if operation == "analyze":
                found_groups = {}
                for group_name, smarts in functional_groups.items():
                    try:
                        pattern = Chem.MolFromSmarts(smarts)
                        if pattern:
                            matches = mol.GetSubstructMatches(pattern)
                            if matches:
                                found_groups[group_name] = {
                                    "count": len(matches),
                                    "smarts": smarts
                                }
                    except Exception:
                        # Skip problematic SMARTS patterns
                        continue

                return json.dumps({
                    "smiles": smiles,
                    "functional_groups": found_groups,
                    "total_functional_groups": len(found_groups)
                })

            elif operation == "suggest_additions":
                # Suggest groups that could improve properties
                suggestions = {
                    "for_solubility": ["primary_alcohol", "carboxylic_acid", "primary_amine"],
                    "for_reduced_toxicity": ["ester", "amide", "primary_alcohol"],
                    "for_drug_likeness": ["primary_alcohol", "secondary_amine", "ester"]
                }
                return json.dumps(suggestions)

            elif operation == "suggest_removals":
                # Suggest problematic groups to remove
                current_groups = []
                for group_name, smarts in functional_groups.items():
                    try:
                        pattern = Chem.MolFromSmarts(smarts)
                        if pattern and mol.GetSubstructMatches(pattern):
                            current_groups.append(group_name)
                    except Exception:
                        continue

                problematic = {
                    "high_toxicity_risk": [g for g in current_groups if g in ["nitro", "halogen", "aldehyde"]],
                    "poor_solubility": [g for g in current_groups if g in ["benzene", "trifluoromethyl"]],
                    "metabolic_liability": [g for g in current_groups if g in ["tertiary_amine", "ester"]]
                }

                return json.dumps({
                    "smiles": smiles,
                    "problematic_groups": problematic
                })

        except Exception as e:
            return json.dumps({"error": str(e)})


class BioisostereGenerator(BaseTool):
    name: str = "Bioisostere Generator"
    description: str = "Generate bioisosteric replacements for functional groups to improve properties."

    def _run(self, smiles: str, target_group: str = ""):
        try:
            # Common bioisosteric replacements
            bioisosteres = {
                "carboxylic_acid": [
                    "tetrazole: c1nnn[nH]1",
                    "hydroxamic_acid: [CX3](=O)[NX3][OX2H]",
                    "acylsulfonamide: [SX4](=O)(=O)[NX3H][CX3]=O"
                ],
                "ester": [
                    "amide: [CX3](=O)[NX3]",
                    "ketone: [CX3]=[OX1]",
                    "ether: [OD2]([#6])[#6]"
                ],
                "benzene": [
                    "pyridine: c1ccncc1",
                    "thiophene: c1ccsc1",
                    "pyrimidine: c1cncnc1"
                ],
                "amide": [
                    "ester: [CX3](=O)[OX2H0]",
                    "ketone: [CX3]=[OX1]",
                    "sulfonamide: [SX4](=O)(=O)[NX3]"
                ]
            }

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return json.dumps({"error": "Invalid SMILES"})

            if target_group and target_group in bioisosteres:
                return json.dumps({
                    "smiles": smiles,
                    "target_group": target_group,
                    "bioisosteric_replacements": bioisosteres[target_group],
                    "recommendation": f"Consider replacing {target_group} with suggested bioisosteres"
                })
            else:
                return json.dumps({
                    "smiles": smiles,
                    "available_bioisosteres": list(bioisosteres.keys()),
                    "all_replacements": bioisosteres
                })

        except Exception as e:
            return json.dumps({"error": str(e)})
