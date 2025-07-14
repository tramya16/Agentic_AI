import pytest
from tools.calculate_synthetic_accessibility import CalculateSA

def test_calculate_sa_valid_smiles():
    tool = CalculateSA()
    smiles = "CCO"  # ethanol, simple valid molecule
    score = tool._run(smiles)
    assert isinstance(score, float)
    assert 1.0 <= score <= 10.0  # SA scores typically between 1 (easy) and 10 (hard)

def test_calculate_sa_invalid_smiles():
    tool = CalculateSA()
    smiles = "INVALIDSMILES"
    with pytest.raises(ValueError):
        tool._run(smiles)

import asyncio

@pytest.mark.asyncio
async def test_calculate_sa_arun_not_implemented():
    tool = CalculateSA()
    with pytest.raises(NotImplementedError):
        await tool._arun("CCO")
