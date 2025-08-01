import pytest
import json
from tools.universal_converter_tool import UniversalConverterTool

tool = UniversalConverterTool()

def parse_output(output: str):
    return json.loads(output)


def test_name_to_smiles():
    result = tool._run("aspirin", input_type="name", output_type="smiles")
    parsed = parse_output(result)
    assert parsed["status"] == "success"
    assert isinstance(parsed["output"], str)
    assert "C" in parsed["output"]  # crude check for SMILES content


def test_smiles_to_inchi():
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    result = tool._run(aspirin_smiles, input_type="smiles", output_type="inchi")
    parsed = parse_output(result)
    assert parsed["status"] == "success"
    assert parsed["output"].startswith("InChI=1S/")


def test_smiles_to_selfies():
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    result = tool._run(aspirin_smiles, input_type="smiles", output_type="selfies")
    print("Resolved SELFIES:", result)
    parsed = parse_output(result)
    assert parsed["status"] == "success"
    assert isinstance(parsed["output"], str)
    assert "[C]" in parsed["output"]  # basic check for SELFIES format


def test_smiles_to_iupac_name():
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    result = tool._run(aspirin_smiles, input_type="smiles", output_type="iupac_name")
    print("Compound fetched:", result)
    parsed = parse_output(result)
    assert parsed["status"] == "success"
    assert isinstance(parsed["output"], str)
    assert "acid" in parsed["output"].lower()


def test_smiles_to_common_name():
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    result = tool._run(aspirin_smiles, input_type="smiles", output_type="common_name")
    parsed = parse_output(result)
    assert parsed["status"] == "success"
    assert isinstance(parsed["output"], str)
    assert "aspirin" in parsed["output"].lower()


def test_invalid_smiles_to_inchi():
    result = tool._run("not-a-smiles", input_type="smiles", output_type="inchi")
    parsed = parse_output(result)
    assert parsed["status"] == "error"


def test_unsupported_output_type():
    result = tool._run("aspirin", input_type="name", output_type="foo")
    parsed = parse_output(result)
    assert parsed["status"] == "error"
    assert "Unsupported output_type" in parsed["error"]
