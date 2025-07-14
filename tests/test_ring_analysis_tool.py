import json
import pytest
from tools.ring_analysis_tool import RingAnalysisTool

def test_ring_analysis_valid_smiles():
    tool = RingAnalysisTool()
    smiles = "c1ccccc1"  # Benzene

    result_json = tool._run(smiles)
    result = json.loads(result_json)

    # Expected keys: ring_count, ring_sizes, spiro_atoms, macrocycles
    assert "ring_count" in result
    assert "ring_sizes" in result
    assert "spiro_atoms_count" in result
    assert "macrocycles_count" in result

    assert isinstance(result["ring_count"], int)
    assert isinstance(result["ring_sizes"], list)
    assert isinstance(result["spiro_atoms_count"], int)
    assert isinstance(result["macrocycles_count"], int)

def test_ring_analysis_invalid_smiles():
    tool = RingAnalysisTool()
    invalid_smiles = "123XYZ"

    result_json = tool._run(invalid_smiles)
    result = json.loads(result_json)

    assert "error" in result
