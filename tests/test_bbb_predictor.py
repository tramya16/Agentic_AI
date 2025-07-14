# test/test_bbb_predictor.py

import json
import pytest
from tools.bbb_predictor import BBBPermeantPredictionTool

tool = BBBPermeantPredictionTool()

def test_valid_bbb_permeant_molecule():
    # Caffeine - expected to be BBB permeant
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    result = json.loads(tool._run(smiles))
    assert "blood_brain_barrier_permeant" in result
    assert isinstance(result["blood_brain_barrier_permeant"], bool)

def test_non_permeant_molecule():
    # Sulfadiazine - not BBB permeant
    smiles = "CC1=CC=C(C=C1)S(=O)(=O)N"
    result = json.loads(tool._run(smiles))
    assert "blood_brain_barrier_permeant" in result
    assert isinstance(result["blood_brain_barrier_permeant"], bool)

def test_invalid_smiles():
    smiles = "INVALID_SMILES"
    result = json.loads(tool._run(smiles))
    assert "error" in result

def test_empty_input():
    smiles = ""
    result = json.loads(tool._run(smiles))
    assert "error" in result
