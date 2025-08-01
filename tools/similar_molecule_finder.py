# tools/similar_molecule_finder.py
import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Import the function from your chemistry utils
try:
    from utils.chemistry_utils import enhanced_find_similar_molecules
except ImportError:
    # Fallback to the original function if enhanced version not available
    from utils.chemistry_utils import find_similar_molecules as enhanced_find_similar_molecules

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
        le=20
    )
    use_chembl: bool = Field(
        True,
        description="Whether to search ChEMBL database"
    )
    use_pubchem: bool = Field(
        True,
        description="Whether to search PubChem database"
    )

class SimilarMoleculeFinderTool(BaseTool):
    name: str = "similar_molecule_finder"
    description: str = (
        "Find structurally similar molecules using multiple databases (PubChem and ChEMBL). "
        "Uses Tanimoto similarity with Morgan fingerprints. Returns CID/ChEMBL ID, SMILES, "
        "name, similarity score, and data source. Enhanced version includes substructure "
        "search fallback and better error handling."
    )
    args_schema: Type[BaseModel] = SimilarMoleculeInput

    def _run(
        self,
        smiles: str,
        similarity_threshold: float = 0.7,
        max_results: int = 5,
        use_chembl: bool = True,
        use_pubchem: bool = True
    ) -> str:
        try:
            # Call the enhanced function with all parameters
            result = enhanced_find_similar_molecules(
                reference_smiles=smiles,
                similarity_threshold=similarity_threshold,
                max_results=max_results,
                use_chembl=use_chembl,
                use_pubchem=use_pubchem
            )
            
            # Format the result for better readability
            if result.get("status") == "success":
                formatted_result = {
                    "status": "success",
                    "reference_smiles": result.get("reference_smiles"),
                    "similarity_threshold": result.get("similarity_threshold"),
                    "total_found": result.get("total_found", 0),
                    "results_returned": len(result.get("results", [])),
                    "sources_used": [src for src in result.get("sources_used", []) if src],
                    "search_strategies": result.get("search_strategies", []),
                    "similar_molecules": []
                }
                
                # Format each result
                for mol in result.get("results", []):
                    formatted_mol = {
                        "similarity_score": mol.get("similarity", 0),
                        "smiles": mol.get("smiles"),
                        "name": mol.get("name", "N/A"),
                        "source": mol.get("source"),
                    }
                    
                    # Add database-specific IDs
                    if "cid" in mol:
                        formatted_mol["pubchem_cid"] = mol["cid"]
                    if "chembl_id" in mol:
                        formatted_mol["chembl_id"] = mol["chembl_id"]
                    
                    formatted_result["similar_molecules"].append(formatted_mol)
                
                return json.dumps(formatted_result, indent=2)
            else:
                return json.dumps(result, indent=2)
                
        except Exception as e:
            error_result = {
                "status": "error",
                "error": f"Tool execution failed: {str(e)}",
                "input_smiles": smiles
            }
            return json.dumps(error_result, indent=2)
