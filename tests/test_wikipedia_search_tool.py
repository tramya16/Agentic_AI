import json
import pytest
from tools.wikipedia_search_tool import WikipediaSearchTool


def test_wikipedia_search_valid_term():
    tool = WikipediaSearchTool()
    query = "Aspirin"
    result = json.loads(tool._run(query))

    assert "summary" in result
    assert "aspirin" in result["summary"].lower()  # Check content relevance
    assert result["query"] == "Aspirin"


def test_wikipedia_search_invalid_term():
    tool = WikipediaSearchTool()
    query = "asdjflkjasdf"  # gibberish
    result = json.loads(tool._run(query))

    assert "summary" in result
    assert "error" in result["summary"].lower() or len(result["summary"]) == 0
