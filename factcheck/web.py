"""Web evidence retrieval via search APIs."""

import os
import requests
from typing import Optional


def search_brave(query: str, count: int = 5, api_key: str = None) -> list[dict]:
    """Search via Brave Search API."""
    key = api_key or os.environ.get("BRAVE_API_KEY", "")
    if not key:
        return []

    headers = {"Accept": "application/json", "X-Subscription-Token": key}
    params = {"q": query, "count": count}
    try:
        resp = requests.get("https://api.search.brave.com/res/v1/web/search",
                          headers=headers, params=params, timeout=15)
        data = resp.json()
        results = []
        for r in data.get("web", {}).get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("description", ""),
            })
        return results
    except Exception:
        return []


def search_duckduckgo(query: str, count: int = 5) -> list[dict]:
    """Fallback: search via DuckDuckGo instant answers (no API key needed)."""
    try:
        resp = requests.get("https://api.duckduckgo.com/",
                          params={"q": query, "format": "json", "no_html": 1},
                          timeout=10,
                          headers={"User-Agent": "FactCheck/0.1"})
        data = resp.json()
        results = []

        # Abstract
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", query),
                "url": data.get("AbstractURL", ""),
                "snippet": data["Abstract"],
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:count]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })

        return results[:count]
    except Exception:
        return []


def fetch_page_text(url: str, max_chars: int = 3000) -> Optional[str]:
    """Fetch and extract readable text from a URL."""
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "FactCheck/0.1 (fact-checking research tool)"
        })
        # Simple HTML to text
        import re
        text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars] if text else None
    except Exception:
        return None


def retrieve_web_evidence(claim: str, api_key: str = None) -> dict:
    """Full web retrieval pipeline: search → fetch top results → extract snippets."""
    # Try Brave first, fallback to DuckDuckGo
    results = search_brave(claim, count=5, api_key=api_key)
    source = "brave"
    if not results:
        results = search_duckduckgo(claim, count=5)
        source = "duckduckgo"

    evidence = {
        "source": source,
        "results": results,
        "pages": [],
    }

    # Fetch top 2 pages for deeper evidence
    for r in results[:2]:
        if r.get("url"):
            text = fetch_page_text(r["url"])
            if text:
                evidence["pages"].append({
                    "url": r["url"],
                    "title": r.get("title", ""),
                    "text": text[:2000],
                })

    return evidence
