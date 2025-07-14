import json
import pytest
from tools.similarity_search_tool import SimilaritySearchTool, SimilarityInput

tool = SimilaritySearchTool()

@pytest.mark.parametrize("fingerprint_type,metric,expected_min", [
    ("morgan", "jaccard", 0.3),
    ("morgan", "dice", 0.3),
    ("morgan", "cosine", 0.3),
    ("maccs", "jaccard", 0.2),
    ("maccs", "dice", 0.2),
    ("maccs", "cosine", 0.2),
    ("rdk", "jaccard", 0.2),
    ("rdk", "dice", 0.2),
    ("rdk", "cosine", 0.2),
])
def test_similarity_success(fingerprint_type, metric, expected_min):
    result_json = tool._run(
        reference_smiles="CCO",
        query_smiles="CCCO",
        fingerprint_type=fingerprint_type,
        metric=metric
    )
    result = json.loads(result_json)

    if result["status"] == "success":
        assert 0.0 <= result["similarity"] <= 1.0
        assert result["similarity"] >= expected_min
    else:
        assert "error" in result
        assert "zero" in result["error"].lower() or "invalid" in result["error"].lower()


def test_invalid_smiles():
    result = tool._run("INVALID", "CCCO", "morgan", "jaccard")
    data = json.loads(result)
    assert data["status"] == "error"
    assert "Invalid SMILES" in data["error"]

def test_unsupported_fingerprint():
    result = tool._run("CCO", "CCCO", "invalid_fp", "jaccard")
    data = json.loads(result)
    assert data["status"] == "error"
    assert "Unsupported fingerprint type" in data["error"]

def test_unsupported_metric():
    result = tool._run("CCO", "CCCO", "morgan", "unknown_metric")
    data = json.loads(result)
    assert data["status"] == "error"
    assert "Unsupported similarity metric" in data["error"]
