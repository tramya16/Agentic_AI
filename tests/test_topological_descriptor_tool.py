import json
import pytest
from tools.topological_descriptor_tool import TopologicalDescriptorTool

def test_topological_descriptor_valid_smiles():
    tool = TopologicalDescriptorTool()
    result = json.loads(tool._run("c1ccccc1"))  # benzene
    assert "chi0" in result
    assert "kappa1" in result
    assert isinstance(result["chi1"], float)

def test_topological_descriptor_invalid_smiles():
    tool = TopologicalDescriptorTool()
    result = json.loads(tool._run("invalid"))
    assert "error" in result
