import pytest
import json
from tools.selfies_converter_tool import SelfiesConverterTool

tool = SelfiesConverterTool()

def test_smiles_to_selfies():
    smiles = "CCO"  # ethanol
    output_json = tool._run(smiles, input_type="smiles")
    output = json.loads(output_json)
    assert output["status"] == "success"
    assert isinstance(output.get("output"), str)
    assert output["output"].startswith("[")  # SELFIES strings start with brackets

def test_selfies_to_smiles():
    selfies = "[C][C][O]"
    output_json = tool._run(selfies, input_type="selfies")
    output = json.loads(output_json)
    assert output["status"] == "success"
    assert isinstance(output.get("smiles"), str)
    assert output["smiles"].startswith("CCO")  # ethanol smiles start

def test_invalid_input_type():
    output_json = tool._run("CCO", input_type="invalid")
    output = json.loads(output_json)
    assert output["status"] == "error"
    assert "Invalid input_type" in output["error"]

def test_invalid_smiles():
    invalid_smiles = "C1CC1C"  # this is valid but let's test error handling with garbage string
    garbage = "XYZ123"
    output_json = tool._run(garbage, input_type="smiles")
    output = json.loads(output_json)
    assert output["status"] == "error"

def test_invalid_selfies():
    invalid_selfies = "[C][Invalid][O]"
    output_json = tool._run(invalid_selfies, input_type="selfies")
    output = json.loads(output_json)
    assert output["status"] == "error"
