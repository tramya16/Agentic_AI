import json
import pytest
from tools.electronic_descriptor_tool import ElectronicDescriptorTool

def test_electronic_descriptor_valid_smiles():
    tool = ElectronicDescriptorTool()
    result = json.loads(tool._run("CC(=O)OC1=CC=CC=C1C(=O)O"))  # Aspirin
    assert "tpsa" in result
    assert result["min_gasteiger_charge"] is not None

def test_electronic_descriptor_invalid_smiles():
    tool = ElectronicDescriptorTool()
    result = json.loads(tool._run("not_a_smiles"))
    assert "error" in result
