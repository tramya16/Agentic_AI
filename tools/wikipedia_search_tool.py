# tools/wikipedia_search_tool.py
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from utils.wiki_utils import fetch_wikipedia_summary
import json

class WikipediaSearchInput(BaseModel):
    query: str = Field(..., description="Term to look up on Wikipedia")

class WikipediaSearchTool(BaseTool):
    name:str = "wikipedia_search"
    description:str = "Fetch and summarize Wikipedia page for a molecule or chemical term"
    args_schema: Type[BaseModel] = WikipediaSearchInput

    def _run(self, query: str) -> str:
        summary = fetch_wikipedia_summary(query)
        return json.dumps({"query": query, "summary": summary})
