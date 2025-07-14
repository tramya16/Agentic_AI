import json
from tools.duplicate_check_tool import DuplicateCheckTool

def test_identical_smiles():
    tool = DuplicateCheckTool()
    result = json.loads(tool._run("CCO", "CCO"))
    assert result["is_duplicate"]

def test_equivalent_smiles_different_order():
    tool = DuplicateCheckTool()
    result = json.loads(tool._run("CCO", "OCC"))
    assert result["is_duplicate"]

def test_non_duplicate_smiles():
    tool = DuplicateCheckTool()
    result = json.loads(tool._run("CCO", "CCC"))
    assert not result["is_duplicate"]
