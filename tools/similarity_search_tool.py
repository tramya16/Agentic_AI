from typing import Type, Literal
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
from rdkit import DataStructs
from scipy.spatial.distance import cosine
import json


class SimilarityInput(BaseModel):
    reference_smiles: str = Field(..., description="Valid SMILES for reference compound")
    query_smiles: str = Field(..., description="Valid SMILES for query compound")
    fingerprint_type: Literal["morgan", "maccs", "rdk"] = Field(
        "morgan", description="Fingerprint type: 'morgan', 'maccs', or 'rdk'"
    )
    metric: Literal["jaccard", "dice", "cosine"] = Field(
        "jaccard", description="Similarity metric to compute"
    )


class SimilaritySearchTool(BaseTool):
    name: str = "similarity_search_tool"
    description: str = "Compute similarity between two molecules using selected fingerprint and metric"
    args_schema: Type[BaseModel] = SimilarityInput

    def _run(self, reference_smiles: str, query_smiles: str,
             fingerprint_type: str = "morgan", metric: str = "jaccard") -> str:
        try:
            mol1 = Chem.MolFromSmiles(reference_smiles)
            mol2 = Chem.MolFromSmiles(query_smiles)
            if not mol1 or not mol2:
                return json.dumps({"status": "error", "error": "Invalid SMILES input"})

            fp1 = self._get_fingerprint(mol1, fingerprint_type)
            fp2 = self._get_fingerprint(mol2, fingerprint_type)

            similarity = self._calculate_similarity(fp1, fp2, metric)
            return json.dumps({
                "status": "success",
                "reference_smiles": Chem.MolToSmiles(mol1),
                "query_smiles": Chem.MolToSmiles(mol2),
                "fingerprint": fingerprint_type,
                "metric": metric,
                "similarity": round(similarity, 4)
            })

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _get_fingerprint(self, mol, fp_type):
        if fp_type == "morgan":
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        elif fp_type == "maccs":
            return MACCSkeys.GenMACCSKeys(mol)
        elif fp_type == "rdk":
            return RDKFingerprint(mol)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")

    def _calculate_similarity(self, fp1, fp2, metric):
        if metric == "jaccard":
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif metric == "dice":
            return DataStructs.DiceSimilarity(fp1, fp2)
        elif metric == "cosine":
            return 1 - cosine(fp1, fp2)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
