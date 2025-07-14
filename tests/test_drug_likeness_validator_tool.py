import json
from tools.drug_likeness_validator_tool import DrugLikenessValidatorTool

tool = DrugLikenessValidatorTool()

def test_lipinski_pass():
    result = json.loads(tool._run("CCO"))  # Ethanol
    assert result["lipinski_pass"]
    assert result["molecular_weight"] < 500

def test_lipinski_fail_high_mw():
    result = json.loads(tool._run("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"))
    assert not result["lipinski_pass"]
    assert result["molecular_weight"] > 500

def test_veber_pass():
    result = json.loads(tool._run("CCO"))  # Ethanol
    assert result["veber_pass"]
    assert result["rotatable_bonds"] <= 10
    assert result["TPSA"] <= 140

def test_veber_fail():
    # A long flexible chain with high TPSA
    result = json.loads(tool._run("CCCCCCCCCCCCCCCC(=O)NC(CC(=O)O)C(=O)O"))  # Often fails Veber
    assert not result["veber_pass"]

def test_invalid_smiles():
    result = json.loads(tool._run("$$$$$"))
    assert result.get("valid") is False
    assert "error" in result

