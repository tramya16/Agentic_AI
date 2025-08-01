import pytest
import json
from tools.formula_calculator_tool import FormulaCalculatorTool

tool = FormulaCalculatorTool()

def test_valid_smiles():
    smiles = "CCO"  # ethanol
    output_json = tool._run(smiles)
    output = json.loads(output_json)
    assert output["status"] == "success"
    assert "molecular_formula" in output
    assert output["molecular_formula"] == "C2H6O"

def test_invalid_smiles():
    invalid_smiles = "XYZ123"
    output_json = tool._run(invalid_smiles)
    output = json.loads(output_json)
    assert output["status"] == "error"

def test_empty_smiles():
    empty_smiles = ""
    output_json = tool._run(empty_smiles)
    print(output_json)
    output = json.loads(output_json)
    assert output["status"] == "error"
