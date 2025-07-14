# tools/similar_molecule_finder.py
import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from utils.chemistry_utils import find_similar_molecules

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

class SimilarMoleculeFinderTool(BaseTool):
    name: str = "similar_molecule_finder"
    description: str = (
        "Find structurally similar molecules in PubChem using Tanimoto similarity. "
        "Returns CID, SMILES, name, and similarity score."
    )
    args_schema: Type[BaseModel] = SimilarMoleculeInput

    def _run(
        self,
        smiles: str,
        similarity_threshold: float = 0.7,
        max_results: int = 5
    ) -> str:
        result = find_similar_molecules(smiles, similarity_threshold, max_results)
        return json.dumps(result)