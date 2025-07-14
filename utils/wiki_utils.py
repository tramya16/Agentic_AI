import wikipedia

def fetch_wikipedia_summary(term: str) -> str:
    try:
        return wikipedia.summary(term, sentences=3,auto_suggest=False)
    except Exception as e:
        return f"Error fetching summary: {e}"
