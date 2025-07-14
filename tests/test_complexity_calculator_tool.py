import json
import pytest
from tools.complexity_calculator_tool import ComplexityCalculatorTool

def test_complexity_calculator_valid_smiles():
    tool = ComplexityCalculatorTool()
    smiles = "CCO"  # Ethanol

    result_json = tool._run(smiles)
    result = json.loads(result_json)

    # Expected keys: bertz_complexity, rotatable_bonds, h_bond_donors, h_bond_acceptors
    assert "bertz_complexity" in result
    assert "rotatable_bonds" in result
    assert "hbond_donors" in result
    assert "hbond_acceptors" in result

    assert isinstance(result["bertz_complexity"], (int, float))
    assert isinstance(result["rotatable_bonds"], int)
    assert isinstance(result["hbond_donors"], int)
    assert isinstance(result["hbond_acceptors"], int)

def test_complexity_calculator_invalid_smiles():
    tool = ComplexityCalculatorTool()
    invalid_smiles = "!!@@"

    result_json = tool._run(invalid_smiles)
    result = json.loads(result_json)

    assert "error" in result
