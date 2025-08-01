from rdkit import Chem
from crewai.tools import BaseTool
from utils import sascorer

class CalculateSA(BaseTool):
    """
    Calculate Synthetic Accessibility (SA) score for molecules.

    Uses the `sascorer.py` implementation from RDKit Contrib:
    Ertl & Schuffenhauer (2009), Journal of Cheminformatics.

    Source: https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
    Licensed under BSD by RDKit.

    Included here with attribution to original authors.

    This work uses the Synthetic Accessibility (SA) scoring method as described by Ertl and Schuffenhauer (2009).
    The implementation is based on the sascorer.py module from the RDKit open-source cheminformatics library, which encodes the algorithm in Python for practical use.
    """

    name:str = "calculate_sa"
    description:str = "Calculate synthetic accessibility (SA) score for a molecule."

    def _run(self, compound: str) -> float:
        mol = Chem.MolFromSmiles(compound)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        return sascorer.calculateScore(mol)

    async def _arun(self, compound: str) -> float:
        raise NotImplementedError()
