
"""mcp_server.py — a 3-tool MCP server for climate-displacement research.
Run standalone:  python mcp_server.py
Inspect:         npx @modelcontextprotocol/inspector python mcp_server.py
"""
from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("research-tools")

# ── A tiny local corpus so recall_memory works offline ──
CORPUS = {
    "unhcr_2023": "According to UNHCR, 21.5 million people are displaced every year by "
                  "sudden climate events: floods, storms and droughts.",
    "worldbank_2050": "The World Bank estimates that by 2050, 216 million people could be "
                      "forced to migrate within their own country because of climate change.",
    "idmc_philippines": "The IDMC reports the Philippines saw a 15% increase in "
                        "typhoon-related displacement over the last five years.",
    "mekong_delta": "The Mekong Delta risks losing up to 40% of its floodable farmland by 2100.",
    "southeast_asia_exposure": "Bangladesh, Vietnam and the Philippines are among the most "
                               "exposed countries to climate displacement in Southeast Asia.",
}
# In-memory store for findings saved during a session
_STORE = {}


@mcp.tool()
def web_search(query: str, num_results: int = 3) -> str:
    """Search the public web for current information.

    Use when: you need facts, news or citations that are NOT already in memory.
    Do NOT use for: maths, or a topic you already saved with store_finding.
    Returns: a numbered list of results, each with a title and a snippet.
    Example: query="UNHCR climate displacement 2024"
    """
    try:
        # DuckDuckGo Instant Answer API — no key required
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        topics = data.get("RelatedTopics", [])
        results = []
        for t in topics[:num_results]:
            if isinstance(t, dict) and t.get("Text"):
                results.append(t["Text"])
        if not results:
            abstract = data.get("AbstractText", "")
            return abstract or "No results found. Try a broader query or use recall_memory."
        return "\n".join(f"{i+1}. {txt}" for i, txt in enumerate(results))
    except requests.Timeout:
        return "Search timed out. Try recall_memory instead."
    except Exception as e:
        return f"Search error: {e}. Try recall_memory instead."


@mcp.tool()
def recall_memory(query: str) -> str:
    """Retrieve relevant passages from the internal knowledge base.

    Use FIRST, before web_search — it is free and instant.
    Returns: matching passages with their source id, or a message to try web_search.
    Example: query="which countries are most exposed"
    """
    try:
        q = set(query.lower().split())
        scored = []
        for doc_id, text in {**CORPUS, **_STORE}.items():
            overlap = len(q & set(text.lower().split()))
            if overlap:
                scored.append((overlap, doc_id, text))
        scored.sort(reverse=True)
        if not scored:
            return "No relevant memories. Use web_search to find new information."
        return "\n---\n".join(f"[{doc_id}] {text}" for _, doc_id, text in scored[:3])
    except Exception as e:
        return f"Recall error: {e}"


@mcp.tool()
def store_finding(finding: str, source: str) -> str:
    """Save a verified finding to memory so recall_memory can return it later.

    Use after web_search when you find a credible, relevant fact.
    Do NOT store: speculation or unverified claims.
    Returns: a confirmation string.
    Example: finding="Sea levels rose 3.6mm/yr", source="NASA 2023"
    """
    try:
        key = f"finding_{len(_STORE) + 1}"
        _STORE[key] = f"{finding} (source: {source})"
        return f"Stored as {key}: {_STORE[key]}"
    except Exception as e:
        return f"Store error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
