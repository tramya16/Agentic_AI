# tests/test_smiles_validator_tool.py
import json
import pytest
from tools.smiles_validator import SmilesValidatorTool

@pytest.fixture
def validator_tool():
    return SmilesValidatorTool()

def test_valid_smiles(validator_tool):
    smiles = "CCO"  # ethanol
    result_json = validator_tool._run(smiles)
    result = json.loads(result_json)
    assert result.get("valid") is True
    assert result.get("canonical_smiles") == "CCO"

def test_invalid_smiles(validator_tool):
    invalid_smiles = "C1(C"
    result_json = validator_tool._run(invalid_smiles)
    result = json.loads(result_json)
    assert result.get("valid") is False
    assert "error" in result

def test_empty_input(validator_tool):
    empty_smiles = ""
    result_json = validator_tool._run(empty_smiles)
    result = json.loads(result_json)
    assert result.get("valid") is False
    assert "error" in result