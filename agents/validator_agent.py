from rdkit import Chem

class ValidatorAgent:
    """
    Validates and sanitizes SMILES strings.
    """

    def __init__(self):
        pass

    def validate(self, smiles: str) -> bool:
        """
        Returns True if the SMILES parses to a valid RDKit Mol and
        passes minimal chemistry sanity checks.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Example additional check: molecule length
        if mol.GetNumAtoms() < 3:
            return False

        # Check allowed elements (H, C, N, O, F, P, S, Cl, Br, I)
        allowed = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in allowed:
                return False

        return True

    def sanitize(self, smiles: str) -> str:
        """
        Converts a valid SMILES to a canonical form.
        Raises ValueError if invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Cannot sanitize invalid SMILES")
        return Chem.MolToSmiles(mol, canonical=True)
