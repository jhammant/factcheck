"""Web evidence retrieval with coarse-to-fine filtering.

Implements WKGFC's web retrieval module:
- Coarse retrieval via Brave/DuckDuckGo
- Fine-grained LLM consistency filtering
- Web evidence → KG triplet alignment
"""

import os
import re
import json
import requests
from typing import Optional, Callable


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
    """Fallback: DuckDuckGo instant answers API."""
    try:
        resp = requests.get("https://api.duckduckgo.com/",
                          params={"q": query, "format": "json", "no_html": 1},
                          timeout=10,
                          headers={"User-Agent": "FactCheck/0.2"})
        data = resp.json()
        results = []

        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", query),
                "url": data.get("AbstractURL", ""),
                "snippet": data["Abstract"],
            })

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


def search_wikipedia(query: str, count: int = 3) -> list[dict]:
    """Search Wikipedia API directly — more reliable than DDG for factual queries."""
    try:
        # Search for pages
        resp = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": count, "format": "json",
        }, timeout=10, headers={"User-Agent": "FactCheck/0.2"})
        data = resp.json()
        
        results = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
            
            # Get the extract for each result
            ext_resp = requests.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "titles": title, "prop": "extracts",
                "exintro": True, "explaintext": True, "format": "json",
            }, timeout=10, headers={"User-Agent": "FactCheck/0.2"})
            ext_data = ext_resp.json()
            pages = ext_data.get("query", {}).get("pages", {})
            extract = ""
            for page in pages.values():
                extract = page.get("extract", "")[:1500]
            
            results.append({
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "snippet": extract or snippet,
            })
        
        return results
    except Exception:
        return []


def fetch_page_text(url: str, max_chars: int = 3000) -> Optional[str]:
    """Fetch and extract readable text from a URL."""
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "FactCheck/0.2 (fact-checking research tool)"
        })
        text = re.sub(r'<script[^>]*>.*?</script>', '', resp.text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars] if text else None
    except Exception:
        return None


def filter_evidence_with_llm(claim: str, passages: list[dict], llm_fn: Callable) -> list[dict]:
    """Fine-grained LLM filter: score each passage for factual relevance to the claim.
    
    This implements WKGFC's coarse-to-fine filtering pipeline.
    Returns passages with consistency scores.
    """
    if not llm_fn or not passages:
        return passages
    
    filtered = []
    for p in passages[:5]:
        snippet = (p.get("snippet") or "")[:500]
        if not snippet:
            continue
            
        prompt = (
            f'Claim: "{claim}"\n\n'
            f'Passage: "{snippet}"\n\n'
            f"Is this passage factually relevant to verifying the claim? "
            f"Reply with a JSON object: {{\"relevant\": true/false, \"consistency\": \"supports\"/\"contradicts\"/\"neutral\", \"confidence\": 0.0-1.0}}"
        )
        resp = llm_fn(prompt)
        try:
            match = re.search(r'\{.*?\}', resp, re.DOTALL)
            if match:
                score = json.loads(match.group())
                if score.get("relevant", False) or score.get("confidence", 0) > 0.3:
                    p["consistency"] = score.get("consistency", "neutral")
                    p["relevance_score"] = score.get("confidence", 0.5)
                    filtered.append(p)
        except (json.JSONDecodeError, AttributeError):
            # If LLM response is unparseable, include the passage anyway
            p["consistency"] = "unknown"
            p["relevance_score"] = 0.5
            filtered.append(p)
    
    # Sort by relevance score
    filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return filtered


def extract_triplets_from_web(claim: str, passages: list[dict], llm_fn: Callable) -> list[dict]:
    """Convert web passages into knowledge graph triplets aligned with KG schema.
    
    This is WKGFC's web→KG alignment step: converting unstructured web text
    into structured (subject, relation, object) triplets.
    """
    if not llm_fn or not passages:
        return []
    
    all_text = "\n".join([
        f"[{p.get('title', '')}]: {(p.get('snippet') or '')[:300]}"
        for p in passages[:3]
    ])
    
    if not all_text.strip():
        return []
    
    prompt = (
        f'Claim: "{claim}"\n\n'
        f"Web evidence:\n{all_text}\n\n"
        f"Extract factual statements from the web evidence as knowledge graph triplets. "
        f"Each triplet should be (subject, relation, object). Focus on facts relevant to "
        f"verifying the claim.\n\n"
        f"Return a JSON array of objects with keys: subject, relation, object\n"
        f"Example: [{{\"subject\": \"Eiffel Tower\", \"relation\": \"located in\", \"object\": \"Paris\"}}]"
    )
    resp = llm_fn(prompt)
    try:
        match = re.search(r'\[.*\]', resp, re.DOTALL)
        if match:
            triplets = json.loads(match.group())
            return [t for t in triplets if isinstance(t, dict) and "subject" in t]
    except (json.JSONDecodeError, AttributeError):
        pass
    return []


def retrieve_web_evidence(claim: str, api_key: str = None, 
                          llm_fn: Callable = None) -> dict:
    """Full web retrieval pipeline with coarse-to-fine filtering.
    
    Pipeline:
    1. Coarse retrieval: Brave → DuckDuckGo → Wikipedia fallback
    2. Fine filtering: LLM scores each passage for relevance
    3. Triplet extraction: Convert web text to KG-aligned triplets
    4. Page extraction: Fetch full text from top results
    """
    # Stage 1: Coarse retrieval (try multiple sources)
    results = search_brave(claim, count=5, api_key=api_key)
    source = "brave"
    
    if not results:
        results = search_wikipedia(claim, count=3)
        source = "wikipedia"
    
    if not results:
        results = search_duckduckgo(claim, count=5)
        source = "duckduckgo"

    evidence = {
        "source": source,
        "results": results,
        "pages": [],
        "triplets": [],
        "filtered_results": [],
    }

    # Stage 2: Fine-grained LLM filtering
    if llm_fn and results:
        evidence["filtered_results"] = filter_evidence_with_llm(claim, results, llm_fn)
    else:
        evidence["filtered_results"] = results

    # Stage 3: Fetch top pages for deeper evidence
    top_results = evidence["filtered_results"][:2] or results[:2]
    for r in top_results:
        if r.get("url"):
            text = fetch_page_text(r["url"])
            if text:
                evidence["pages"].append({
                    "url": r["url"],
                    "title": r.get("title", ""),
                    "text": text[:2000],
                })

    # Stage 4: Extract triplets from web evidence (web→KG alignment)
    if llm_fn and (evidence["filtered_results"] or evidence["pages"]):
        sources = evidence["filtered_results"] or results
        evidence["triplets"] = extract_triplets_from_web(claim, sources, llm_fn)

    return evidence
