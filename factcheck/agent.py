"""Agentic fact-checking loop — decides what evidence to gather and delivers a verdict."""

import json
import requests
from typing import Callable, Optional

from .kg import retrieve_kg_evidence
from .web import retrieve_web_evidence


VERDICTS = ["SUPPORTED", "REFUTED", "NOT ENOUGH EVIDENCE"]


def call_ollama(prompt: str, model: str = "qwen2.5:14b", base_url: str = "http://localhost:11434") -> str:
    """Call Ollama API for LLM inference."""
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        return resp.json().get("response", "")
    except Exception as e:
        return f"[LLM Error: {e}]"


def call_openai(prompt: str, model: str = "gpt-4o-mini", api_key: str = None, base_url: str = None) -> str:
    """Call OpenAI-compatible API."""
    import os
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    url = base_url or "https://api.openai.com/v1/chat/completions"
    if not key:
        return "[No OpenAI API key]"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 1000},
            timeout=60,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM Error: {e}]"


def call_gemini(prompt: str, model: str = "gemini-2.0-flash", api_key: str = None) -> str:
    """Call Google Gemini API."""
    import os
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return "[No Gemini API key]"
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        resp = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000},
        }, timeout=60)
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[LLM Error: {e}]"


def make_llm_fn(provider: str = "ollama", model: str = None, **kwargs) -> Callable:
    """Create an LLM function based on provider."""
    if provider == "ollama":
        m = model or "qwen2.5:14b"
        return lambda prompt: call_ollama(prompt, model=m, **kwargs)
    elif provider == "openai":
        m = model or "gpt-4o-mini"
        return lambda prompt: call_openai(prompt, model=m, **kwargs)
    elif provider == "gemini":
        m = model or "gemini-2.0-flash"
        return lambda prompt: call_gemini(prompt, model=m, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use: ollama, openai, gemini")


def format_kg_evidence(kg: dict) -> str:
    """Format KG evidence into readable text."""
    lines = []
    if kg.get("entities"):
        lines.append("**Resolved Entities:**")
        for e in kg["entities"]:
            lines.append(f"  - {e['label']} ({e['id']}): {e.get('description', 'N/A')}")

    if kg.get("facts"):
        lines.append("\n**Knowledge Graph Facts:**")
        seen = set()
        for f in kg["facts"]:
            key = f"{f.get('entity', '')}: {f['property']} = {f['value']}"
            if key not in seen:
                seen.add(key)
                lines.append(f"  - [{f.get('entity', '?')}] {f['property']}: {f['value']}")

    if kg.get("expanded"):
        lines.append("\n**Expanded Relations:**")
        for f in kg["expanded"][:10]:
            lines.append(f"  - {f['related_entity']} → {f['property']}: {f['value']}")

    return "\n".join(lines) if lines else "No KG evidence found."


def format_web_evidence(web: dict) -> str:
    """Format web evidence into readable text."""
    lines = []
    if web.get("results"):
        lines.append(f"**Web Search Results ({web.get('source', 'unknown')}):**")
        for r in web["results"]:
            lines.append(f"  - [{r['title']}]({r['url']})")
            if r.get("snippet"):
                lines.append(f"    {r['snippet'][:200]}")

    if web.get("pages"):
        lines.append("\n**Page Extracts:**")
        for p in web["pages"]:
            lines.append(f"  Source: {p['url']}")
            lines.append(f"  {p['text'][:500]}...")

    return "\n".join(lines) if lines else "No web evidence found."


def verify_claim(
    claim: str,
    provider: str = "ollama",
    model: str = None,
    brave_api_key: str = None,
    max_steps: int = 4,
    verbose: bool = False,
) -> dict:
    """
    Agentic fact-checking loop.

    1. Extract entities and search Knowledge Graph
    2. Assess if evidence is sufficient
    3. If not, search the web
    4. Deliver verdict with citations
    """
    llm_fn = make_llm_fn(provider, model)
    steps = []
    kg_evidence = None
    web_evidence = None

    # Step 1: Knowledge Graph retrieval
    if verbose:
        print("🔍 Step 1: Searching Knowledge Graph...")
    kg_evidence = retrieve_kg_evidence(claim, llm_fn)
    kg_text = format_kg_evidence(kg_evidence)
    steps.append({"action": "kg_retrieval", "result": kg_text})

    if verbose:
        print(f"   Found {len(kg_evidence.get('facts', []))} facts, "
              f"{len(kg_evidence.get('entities', []))} entities")

    # Step 2: Ask LLM if KG evidence is sufficient
    sufficiency_prompt = f"""You are a fact-checker. Given this claim and evidence from a knowledge graph, 
is the evidence SUFFICIENT to make a verdict? 

Claim: "{claim}"

Evidence:
{kg_text}

Reply with ONLY one of:
- "SUFFICIENT" if you can confidently verify or refute the claim
- "INSUFFICIENT" if you need more evidence (explain what's missing in one sentence)
"""
    sufficiency = llm_fn(sufficiency_prompt).strip()
    steps.append({"action": "assess_sufficiency", "result": sufficiency})

    # Step 3: Web search if needed
    if "INSUFFICIENT" in sufficiency.upper() or len(kg_evidence.get("facts", [])) < 3:
        if verbose:
            print("🌐 Step 2: KG insufficient, searching web...")
        web_evidence = retrieve_web_evidence(claim, api_key=brave_api_key)
        web_text = format_web_evidence(web_evidence)
        steps.append({"action": "web_retrieval", "result": web_text})

        if verbose:
            print(f"   Found {len(web_evidence.get('results', []))} web results, "
                  f"{len(web_evidence.get('pages', []))} page extracts")
    else:
        web_text = ""
        if verbose:
            print("✅ Step 2: KG evidence sufficient, skipping web search")

    # Step 4: Final verdict
    if verbose:
        print("⚖️  Step 3: Delivering verdict...")

    all_evidence = kg_text
    if web_text:
        all_evidence += "\n\n" + web_text

    verdict_prompt = f"""You are an expert fact-checker. Analyse the claim against ALL available evidence and deliver a verdict.

CLAIM: "{claim}"

EVIDENCE:
{all_evidence}

Instructions:
1. Analyse each piece of evidence for relevance to the claim
2. Identify any contradictions between evidence sources
3. Deliver your verdict as one of: SUPPORTED, REFUTED, or NOT ENOUGH EVIDENCE
4. Provide a clear explanation citing specific evidence
5. Rate your confidence: HIGH, MEDIUM, or LOW

Format your response EXACTLY as:
VERDICT: [SUPPORTED/REFUTED/NOT ENOUGH EVIDENCE]
CONFIDENCE: [HIGH/MEDIUM/LOW]
EXPLANATION: [Your reasoning with citations]
KEY EVIDENCE: [The most important facts that support your verdict]
"""
    verdict_raw = llm_fn(verdict_prompt)
    steps.append({"action": "verdict", "result": verdict_raw})

    # Parse verdict
    verdict = "NOT ENOUGH EVIDENCE"
    confidence = "LOW"
    explanation = verdict_raw

    for v in VERDICTS:
        if v in verdict_raw.upper():
            verdict = v
            break

    for c in ["HIGH", "MEDIUM", "LOW"]:
        if f"CONFIDENCE: {c}" in verdict_raw.upper():
            confidence = c
            break

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "explanation": verdict_raw,
        "steps": steps,
        "evidence": {
            "kg": {
                "entities": len(kg_evidence.get("entities", [])) if kg_evidence else 0,
                "facts": len(kg_evidence.get("facts", [])) if kg_evidence else 0,
            },
            "web": {
                "results": len(web_evidence.get("results", [])) if web_evidence else 0,
                "pages": len(web_evidence.get("pages", [])) if web_evidence else 0,
            },
        },
    }
