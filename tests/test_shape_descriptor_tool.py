import json
import pytest
from tools.shape_descriptor_tool import ShapeDescriptorTool

def test_shape_descriptor_valid_smiles():
    tool = ShapeDescriptorTool()
    result = json.loads(tool._run("CCO"))  # ethanol
    assert "asphericity" in result
    assert isinstance(result["inertial_shape_factor"], float)

def test_shape_descriptor_invalid_smiles():
    tool = ShapeDescriptorTool()
    result = json.loads(tool._run("1234XYZ"))
    assert "error" in result
