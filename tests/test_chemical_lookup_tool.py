# tests/test_chemical_lookup_tool.py
import json
import pytest
from tools.properties_lookup import PropertiesLookupTool

@pytest.fixture
def lookup_tool():
    return PropertiesLookupTool()

def test_valid_smiles(lookup_tool):
    smiles = "CCO"  # ethanol
    result_json = lookup_tool._run(smiles)
    result = json.loads(result_json)
    assert result.get("valid") is True
    assert "molecular_formula" in result
    assert result["molecular_formula"] == "C2H6O"

def test_invalid_smiles(lookup_tool):
    invalid_smiles = "C1(C"  # invalid
    result_json = lookup_tool._run(invalid_smiles)
    result = json.loads(result_json)
    assert result.get("valid") is False
    assert "error" in result
