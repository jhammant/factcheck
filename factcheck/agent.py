"""Agentic fact-checking loop implementing WKGFC-inspired evidence acquisition.

Based on WKGFC (arXiv 2603.00267) with practical adaptations:
- Two modes: "fast" (2 LLM calls, good for small models) and "deep" (MDP agent loop)
- Bidirectional KG retrieval with beam search expansion
- Wikipedia as a reliable web evidence source
- Coarse-to-fine web filtering
- Web→KG triplet alignment
"""

import json
import re
import requests
from typing import Callable, Optional

from .kg import retrieve_kg_evidence, beam_search_expand, resolve_entity, get_entity_facts_bidirectional
from .web import retrieve_web_evidence


VERDICTS = ["SUPPORTED", "REFUTED", "NOT ENOUGH EVIDENCE"]

# ============ LLM Provider Functions ============

def call_ollama(prompt: str, model: str = "qwen2.5:14b", base_url: str = "http://localhost:11434") -> str:
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        return resp.json().get("response", "")
    except Exception as e:
        return f"[LLM Error: {e}]"


def call_openai(prompt: str, model: str = "gpt-4o-mini", api_key: str = None, base_url: str = None) -> str:
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


def call_gemini(prompt: str, model: str = "gemini-2.5-flash", api_key: str = None) -> str:
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
        if "error" in data:
            return f"[Gemini API Error: {data['error'].get('message', data['error'])}]"
        if "candidates" not in data:
            block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
            return f"[Gemini blocked response: {block_reason}]"
        candidate = data["candidates"][0]
        if candidate.get("finishReason") not in (None, "STOP"):
            return f"[Gemini blocked response: {candidate.get('finishReason')}]"
        return candidate["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[LLM Error: {e}]"


def make_llm_fn(provider: str = "ollama", model: str = None, **kwargs) -> Callable:
    if provider == "ollama":
        return lambda prompt: call_ollama(prompt, model=model or "qwen2.5:14b", **kwargs)
    elif provider == "openai":
        return lambda prompt: call_openai(prompt, model=model or "gpt-4o-mini", **kwargs)
    elif provider == "gemini":
        return lambda prompt: call_gemini(prompt, model=model or "gemini-2.5-flash", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use: ollama, openai, gemini")


# ============ Evidence Formatting ============

def format_kg_evidence(kg: dict) -> str:
    lines = []
    if kg.get("entities"):
        lines.append("**Resolved Entities:**")
        for e in kg["entities"]:
            lines.append(f"  - {e['label']} ({e['id']}): {e.get('description', 'N/A')}")

    if kg.get("facts"):
        lines.append("\n**Knowledge Graph Facts:**")
        seen = set()
        for f in kg["facts"]:
            direction = f.get("direction", "outgoing")
            entity = f.get("entity", "?")
            key = f"{entity}: {f['property']} = {f['value']}"
            if key not in seen:
                seen.add(key)
                prefix = "←" if direction == "incoming" else "→"
                lines.append(f"  {prefix} [{entity}] {f['property']}: {f['value']}")

    if kg.get("expanded"):
        lines.append("\n**Expanded Relations (beam search):**")
        for f in kg["expanded"][:15]:
            hop = f.get("hop", "?")
            lines.append(f"  (hop {hop}) {f.get('from', '?')} —[{f['property']}]→ {f['related_entity']}")

    return "\n".join(lines) if lines else "No KG evidence found."


def format_web_evidence(web: dict) -> str:
    lines = []
    results_key = "filtered_results" if web.get("filtered_results") else "results"
    results = web.get(results_key, [])
    
    if results:
        lines.append(f"**Web Evidence ({web.get('source', 'unknown')}):**")
        for r in results[:5]:
            lines.append(f"  - {r.get('title', '?')}")
            if r.get("snippet"):
                lines.append(f"    {r['snippet'][:300]}")

    if web.get("triplets"):
        lines.append("\n**Extracted Facts from Web:**")
        for t in web["triplets"][:10]:
            lines.append(f"  ({t.get('subject', '?')}, {t.get('relation', '?')}, {t.get('object', '?')})")

    if web.get("pages"):
        lines.append("\n**Page Extracts:**")
        for p in web["pages"][:2]:
            lines.append(f"  Source: {p['url']}")
            lines.append(f"  {p['text'][:400]}...")

    return "\n".join(lines) if lines else "No web evidence found."


# ============ Fact Checking Modes ============

def _assess_kg_sufficiency(kg_evidence: dict, claim: str) -> str:
    """Quick heuristic check — does the KG evidence address the claim?
    Returns: 'sufficient', 'weak', or 'none'"""
    n_facts = len(kg_evidence.get("facts", []))
    n_entities = len(kg_evidence.get("entities", []))
    n_expanded = len(kg_evidence.get("expanded", []))
    
    if n_entities == 0:
        return "none"
    if n_facts < 3 and n_expanded < 2:
        return "weak"
    if n_facts >= 5:
        return "sufficient"
    return "weak"


def verify_claim(
    claim: str,
    provider: str = "ollama",
    model: str = None,
    brave_api_key: str = None,
    max_steps: int = 4,
    beam_width: int = 5,
    max_hops: int = 2,
    mode: str = "fast",
    verbose: bool = False,
) -> dict:
    """
    Fact-check a claim using Knowledge Graphs + Web Search + LLM.
    
    Modes:
    - "fast": 2 LLM calls (entity extraction + verdict). Good for small/slow models.
              KG retrieval uses heuristic beam search, web search always runs.
    - "deep": Full MDP agent loop. Agent decides expandKG/webSearch/verdict at each step.
              Uses LLM for entity disambiguation, beam pruning, web filtering, triplet extraction.
              Needs a capable model (7B+ recommended).
    
    The fast mode still uses bidirectional SPARQL and beam search expansion,
    just without LLM-guided pruning at each step.
    """
    llm_fn = make_llm_fn(provider, model)
    steps = []
    kg_evidence = None
    web_evidence = None

    # In fast mode, don't pass llm_fn to KG retrieval (avoids extra LLM calls)
    kg_llm = llm_fn if mode == "deep" else None

    # ──── Step 1: KG Retrieval ────
    if verbose:
        print(f"🔍 KG Retrieval (mode={mode}, hops={max_hops}, beam={beam_width})...")
    
    kg_evidence = retrieve_kg_evidence(
        claim, llm_fn=kg_llm, 
        max_hops=max_hops, beam_width=beam_width
    )
    kg_text = format_kg_evidence(kg_evidence)
    steps.append({"action": "initKGRetrieval", "result": kg_text})

    if verbose:
        n_facts = len(kg_evidence.get("facts", []))
        n_ents = len(kg_evidence.get("entities", []))
        n_exp = len(kg_evidence.get("expanded", []))
        print(f"   {n_ents} entities, {n_facts} facts, {n_exp} expanded relations")

    # ──── Step 2: Decide if we need web evidence ────
    kg_quality = _assess_kg_sufficiency(kg_evidence, claim)
    
    if mode == "fast":
        # Fast mode: always fetch web evidence for completeness
        # Wikipedia is free and fast — no reason not to
        if verbose:
            print(f"🌐 Web Search (KG quality: {kg_quality})...")
        web_evidence = retrieve_web_evidence(claim, api_key=brave_api_key, llm_fn=None)
        web_text = format_web_evidence(web_evidence)
        steps.append({"action": "webSearch", "result": web_text})
        
        if verbose:
            src = web_evidence.get("source", "?")
            n_results = len(web_evidence.get("results", []))
            print(f"   {n_results} results from {src}")
    
    elif mode == "deep":
        web_text = ""
        # Deep mode: MDP agent loop
        for step_num in range(2, max_steps + 1):
            # Agent decides action
            action_prompt = f"""You are a fact-checking agent. Given this claim and evidence, what should you do next?

CLAIM: "{claim}"
KG EVIDENCE QUALITY: {kg_quality}
{kg_text[:1500]}
{web_text[:500] if web_text else "(No web evidence yet)"}

Choose ONE action:
- expandKG — need more knowledge graph relations
- webSearch — need web sources to complement KG
- verdict — have enough evidence to decide

Reply with ONLY the action name."""
            
            action = llm_fn(action_prompt).strip().lower()
            
            if "expandkg" in action and kg_evidence.get("entities"):
                if verbose:
                    print(f"🔄 Step {step_num}: expandKG")
                for entity in kg_evidence["entities"][1:3]:
                    extra = beam_search_expand(
                        entity["id"], entity["label"], claim,
                        llm_fn=llm_fn, beam_width=3, max_hops=1,
                    )
                    kg_evidence["expanded"].extend(extra)
                kg_text = format_kg_evidence(kg_evidence)
                kg_quality = _assess_kg_sufficiency(kg_evidence, claim)
                steps.append({"action": "expandKG", "result": f"Expanded: {len(kg_evidence['facts'])} facts"})
                
            elif "websearch" in action:
                if verbose:
                    print(f"🌐 Step {step_num}: webSearch")
                web_evidence = retrieve_web_evidence(claim, api_key=brave_api_key, llm_fn=llm_fn)
                web_text = format_web_evidence(web_evidence)
                steps.append({"action": "webSearch", "result": web_text})
            else:
                if verbose:
                    print(f"⚖️  Step {step_num}: verdict")
                break
    else:
        web_text = ""

    # ──── Final Step: Verdict ────
    if verbose:
        print("⚖️  Delivering verdict...")

    all_evidence = kg_text
    if web_text:
        all_evidence += "\n\n" + web_text

    verdict_prompt = f"""You are a fact-checker. Analyse the claim against the evidence and deliver a verdict.

CLAIM: "{claim}"

EVIDENCE:
{all_evidence}

RULES:
- Compare the SPECIFIC claim against the evidence
- If evidence shows a DIFFERENT answer than what the claim states → REFUTED
  Example: Claim says "capital is Sydney" but evidence shows capital is Canberra → REFUTED
  Example: Claim says "founded by X" but evidence shows founded by Y, X joined later → REFUTED
  Example: Claim says "happened in 1991" but evidence shows 1989 → REFUTED
- If evidence confirms the claim → SUPPORTED
- Only say NOT ENOUGH EVIDENCE if genuinely unable to determine
- Pay attention to incoming relations (←) which show what points TO an entity

Format EXACTLY as:
VERDICT: [SUPPORTED/REFUTED/NOT ENOUGH EVIDENCE]
CONFIDENCE: [HIGH/MEDIUM/LOW]
EXPLANATION: [Your reasoning citing specific evidence]
KEY EVIDENCE: [2-3 most important facts]
"""
    verdict_raw = llm_fn(verdict_prompt)
    steps.append({"action": "verdict", "result": verdict_raw})

    # Parse verdict
    verdict = "NOT ENOUGH EVIDENCE"
    confidence = "LOW"

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
                "expanded": len(kg_evidence.get("expanded", [])) if kg_evidence else 0,
            },
            "web": {
                "results": len(web_evidence.get("results", [])) if web_evidence else 0,
                "pages": len(web_evidence.get("pages", [])) if web_evidence else 0,
                "triplets": len(web_evidence.get("triplets", [])) if web_evidence else 0,
            },
        },
    }
