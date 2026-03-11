"""Knowledge Graph retrieval via Wikidata SPARQL endpoint.

Implements expand-and-prune beam search as described in WKGFC (arXiv 2603.00267).
Key improvements over v1:
- Bidirectional SPARQL (incoming + outgoing relations)
- LLM-guided beam search with relevance pruning
- Multi-hop expansion (configurable depth)
- Better entity resolution with LLM disambiguation
"""

import re
import json
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Callable, Optional


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
sparql.setReturnFormat(JSON)
sparql.addCustomHttpHeader("User-Agent", "FactCheck/0.2 (https://github.com/jhammant/factcheck)")


# Properties to always skip (IDs, coordinates, media files)
SKIP_PROPERTIES = {
    "freebase id", "mesh descriptor id", "mesh tree code", "gnd id", "viaf id",
    "library of congress authority id", "bnf id", "imdb id", "geonames id",
    "openstreetmap relation id", "iso 3166-2 code", "local dialing code",
    "postal code", "coordinate location", "commons category", "topic's main category",
    "page banner", "image", "flag image", "coat of arms image", "locator map image",
    "pronunciation audio", "spoken text audio", "nl cr aut id", "nuts code",
}


def extract_entities(text: str, llm_fn: Optional[Callable] = None) -> list[str]:
    """Extract entity names from a claim. Uses LLM if available, else regex NER."""
    if llm_fn:
        prompt = (
            "Extract ALL named entities (people, places, organisations, events, things, concepts) "
            "from this claim. Include proper nouns AND important common nouns that could be looked up "
            "in a knowledge base (e.g. 'Nobel Prize', 'World War II', 'DNA').\n"
            "Return ONLY a JSON array of strings, nothing else.\n\n"
            f'Claim: "{text}"\n\nEntities:'
        )
        resp = llm_fn(prompt)
        try:
            match = re.search(r'\[.*?\]', resp, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                if entities and len(entities) > 0:
                    return [str(e) for e in entities]
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: capitalised word sequences
    entities = re.findall(r'(?:[A-Z][a-z]+(?:\s+(?:the\s+)?[A-Z][a-z]+)*)', text)
    the_entities = re.findall(r'[Tt]he\s+((?:[A-Z][a-z]+\s*)+)', text)
    entities.extend([e.strip() for e in the_entities])

    stop = {"This", "That", "These", "Those", "There", "When", "Where",
            "What", "Which", "How", "Who", "Its", "His", "Her", "Did", "Was",
            "Were", "Has", "Have", "Had", "Are", "Been", "Being", "Does", "Do",
            "Located", "Born", "Died", "Known", "Won", "Made", "Called", "Also"}

    seen = set()
    result = []
    for e in entities:
        if e not in stop and len(e) > 1 and e.lower() not in seen:
            seen.add(e.lower())
            result.append(e)
    return result


def resolve_entity(name: str, claim: str = "", llm_fn: Optional[Callable] = None) -> dict | None:
    """Resolve entity name to Wikidata item. Uses LLM for disambiguation if available."""
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "limit": 10,
        "format": "json",
    }
    try:
        resp = requests.get(WIKIDATA_API, params=params, timeout=10,
                          headers={"User-Agent": "FactCheck/0.2"})
        data = resp.json()
        results = data.get("search", [])
        if not results:
            return None

        # If LLM available and multiple candidates, use LLM to disambiguate
        if llm_fn and len(results) > 1 and claim:
            candidates = []
            for i, r in enumerate(results[:5]):
                desc = r.get("description", "no description")
                candidates.append(f"{i+1}. {r.get('label', name)} ({r['id']}): {desc}")

            prompt = (
                f'Given the claim: "{claim}"\n'
                f'Which Wikidata entity best matches "{name}"?\n\n'
                + "\n".join(candidates) + "\n\n"
                "Reply with ONLY the number (1-5)."
            )
            resp_text = llm_fn(prompt).strip()
            try:
                idx = int(re.search(r'\d+', resp_text).group()) - 1
                if 0 <= idx < len(results):
                    r = results[idx]
                    return {
                        "id": r["id"],
                        "label": r.get("label", name),
                        "description": r.get("description", ""),
                    }
            except (ValueError, AttributeError):
                pass

        # Fallback: score-based ranking
        artwork_terms = {"painting", "song", "album", "film", "episode", "novel",
                        "sculpture", "photograph", "artwork", "book by", "poem"}
        best = None
        best_score = -1
        for r in results:
            desc = (r.get("description") or "").lower()
            score = 0
            if any(t in desc for t in ["city", "country", "capital", "state", "continent",
                                        "person", "politician", "scientist", "born",
                                        "company", "organisation", "organization",
                                        "tower", "building", "structure", "landmark",
                                        "river", "mountain", "island", "university",
                                        "chemical", "element", "planet", "moon",
                                        "programming language", "software", "website"]):
                score += 10
            if any(t in desc for t in artwork_terms):
                score -= 5
            if desc:
                score += 1
            qid = r.get("id", "Q999999999")
            try:
                qnum = int(qid[1:])
                if qnum < 1000: score += 5
                elif qnum < 10000: score += 3
                elif qnum < 100000: score += 1
            except ValueError:
                pass
            if score > best_score:
                best_score = score
                best = r

        if best:
            return {
                "id": best["id"],
                "label": best.get("label", name),
                "description": best.get("description", ""),
            }
    except Exception:
        pass
    return None


def _is_useful_value(val: str) -> bool:
    """Filter out URIs, QIDs, coordinates, and pure numeric timestamps."""
    if not val or val.startswith("http") or val.startswith("Point("):
        return False
    if re.match(r'^[QP]\d+$', val):
        return False
    if re.match(r'^[\d/\-T:.]+Z?$', val):
        return False
    return True



# High-value Wikidata properties for fact-checking (property IDs)
# These are queried FIRST to ensure we get the important facts
PRIORITY_PROPERTIES = [
    "P36",   # capital
    "P17",   # country
    "P131",  # located in admin entity
    "P30",   # continent
    "P31",   # instance of
    "P279",  # subclass of
    "P27",   # country of citizenship
    "P19",   # place of birth
    "P20",   # place of death
    "P569",  # date of birth
    "P570",  # date of death
    "P112",  # founded by
    "P571",  # inception (founding date)
    "P127",  # owned by
    "P176",  # manufacturer
    "P138",  # named after
    "P1376", # capital of
    "P37",   # official language
    "P38",   # currency
    "P6",    # head of government
    "P35",   # head of state
    "P166",  # award received
    "P800",  # notable work
    "P106",  # occupation
    "P39",   # position held
    "P108",  # employer
    "P69",   # educated at
    "P26",   # spouse
    "P22",   # father
    "P25",   # mother
    "P40",   # child
    "P175",  # performer
    "P50",   # author
    "P170",  # creator
    "P86",   # composer
    "P577",  # publication date
    "P1082", # population
    "P2044", # elevation
    "P2046", # area
    "P47",   # shares border with
    "P150",  # contains admin entity
    "P361",  # part of
    "P527",  # has part
    "P155",  # follows (predecessor)
    "P156",  # followed by (successor)
]


def get_entity_facts_bidirectional(entity_id: str, max_facts: int = 30) -> list[dict]:
    """Retrieve facts about an entity — BOTH outgoing and incoming relations.
    
    Uses a two-pass strategy:
    1. First pass: query priority properties (capital, founded by, etc.)
    2. Second pass: general query for remaining facts
    
    This ensures claim-relevant properties aren't drowned out by noise.
    """
    # Pass 1: Priority properties (these are most useful for fact-checking)
    prop_filter = " ".join([f"wdt:{p}" for p in PRIORITY_PROPERTIES[:20]])
    priority_query = f"""
    SELECT ?propertyLabel ?valueLabel WHERE {{
      wd:{entity_id} ?directProp ?value .
      ?property wikibase:directClaim ?directProp .
      FILTER(?directProp IN ({", ".join([f"wdt:{p}" for p in PRIORITY_PROPERTIES])}))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT {max_facts}
    """

    # Pass 2: General outgoing (for remaining slots)
    outgoing_query = f"""
    SELECT ?propertyLabel ?valueLabel WHERE {{
      wd:{entity_id} ?directProp ?value .
      ?property wikibase:directClaim ?directProp .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT {max_facts}
    """

    # Incoming: what points TO this entity
    incoming_query = f"""
    SELECT ?subjectLabel ?propertyLabel WHERE {{
      ?subject ?directProp wd:{entity_id} .
      ?property wikibase:directClaim ?directProp .
      FILTER(?directProp IN ({", ".join([f"wdt:{p}" for p in PRIORITY_PROPERTIES[:15]])}))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT 15
    """

    facts = []
    seen = set()

    for query, direction in [(priority_query, "outgoing"), (outgoing_query, "outgoing"), (incoming_query, "incoming")]:
        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
            for r in results["results"]["bindings"]:
                if direction == "outgoing":
                    prop = r.get("propertyLabel", {}).get("value", "")
                    val = r.get("valueLabel", {}).get("value", "")
                    if not _is_useful_value(val) or prop.lower() in SKIP_PROPERTIES:
                        continue
                    key = f"out:{prop}:{val}"
                    if key not in seen:
                        seen.add(key)
                        facts.append({"property": prop, "value": val, "direction": "outgoing"})
                else:
                    subj = r.get("subjectLabel", {}).get("value", "")
                    prop = r.get("propertyLabel", {}).get("value", "")
                    if not _is_useful_value(subj) or prop.lower() in SKIP_PROPERTIES:
                        continue
                    key = f"in:{subj}:{prop}"
                    if key not in seen:
                        seen.add(key)
                        facts.append({"property": f"{prop} (of {subj})", "value": subj, 
                                     "direction": "incoming"})
        except Exception:
            continue

    return facts


def beam_search_expand(
    entity_id: str,
    entity_label: str,
    claim: str,
    llm_fn: Optional[Callable] = None,
    beam_width: int = 5,
    max_hops: int = 2,
) -> list[dict]:
    """Expand KG via beam search with LLM-guided pruning.
    
    At each hop:
    1. Get all relations from current entity
    2. Use LLM to rank relations by relevance to claim
    3. Keep top-k (beam_width) and follow them
    4. Repeat for max_hops
    
    This is the core of WKGFC's expand-and-prune strategy.
    """
    expanded = []
    current_entities = [(entity_id, entity_label)]
    visited = {entity_id}

    for hop in range(max_hops):
        hop_candidates = []

        for eid, elabel in current_entities:
            # Get outgoing relations with their target entities
            query = f"""
            SELECT ?propLabel ?targetLabel ?target WHERE {{
              wd:{eid} ?directProp ?target .
              ?property wikibase:directClaim ?directProp .
              FILTER(STRSTARTS(STR(?target), "http://www.wikidata.org/entity/Q"))
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
            }}
            LIMIT 30
            """
            sparql.setQuery(query)
            try:
                results = sparql.query().convert()
                for r in results["results"]["bindings"]:
                    prop = r.get("propLabel", {}).get("value", "")
                    target_label = r.get("targetLabel", {}).get("value", "")
                    target_uri = r.get("target", {}).get("value", "")
                    target_id = target_uri.split("/")[-1] if target_uri else ""

                    if (not _is_useful_value(target_label) or 
                        prop.lower() in SKIP_PROPERTIES or
                        target_id in visited):
                        continue

                    hop_candidates.append({
                        "from_entity": elabel,
                        "from_id": eid,
                        "relation": prop,
                        "target_label": target_label,
                        "target_id": target_id,
                    })
            except Exception:
                continue

        if not hop_candidates:
            break

        # LLM-guided pruning: rank candidates by relevance to claim
        if llm_fn and len(hop_candidates) > beam_width:
            candidate_text = "\n".join([
                f"{i+1}. {c['from_entity']} —[{c['relation']}]→ {c['target_label']}"
                for i, c in enumerate(hop_candidates[:20])
            ])
            prompt = (
                f'Claim to verify: "{claim}"\n\n'
                f"Which of these knowledge graph relations are most relevant for "
                f"verifying this claim? List the numbers of the top {beam_width} most "
                f"relevant, separated by commas.\n\n"
                f"{candidate_text}\n\n"
                f"Most relevant (comma-separated numbers):"
            )
            resp = llm_fn(prompt)
            try:
                nums = [int(n.strip()) - 1 for n in re.findall(r'\d+', resp)]
                selected = [hop_candidates[n] for n in nums if 0 <= n < len(hop_candidates)]
                if selected:
                    hop_candidates = selected[:beam_width]
            except (ValueError, IndexError):
                hop_candidates = hop_candidates[:beam_width]
        else:
            hop_candidates = hop_candidates[:beam_width]

        # Add to results and prepare next hop
        next_entities = []
        for c in hop_candidates:
            visited.add(c["target_id"])
            expanded.append({
                "related_entity": c["target_label"],
                "property": c["relation"],
                "value": c["target_label"],
                "from": c["from_entity"],
                "hop": hop + 1,
            })
            next_entities.append((c["target_id"], c["target_label"]))

        current_entities = next_entities

    return expanded


def retrieve_kg_evidence(claim: str, llm_fn: Optional[Callable] = None, 
                         max_hops: int = 2, beam_width: int = 5) -> dict:
    """Full KG retrieval pipeline with beam search expansion.
    
    Pipeline:
    1. Extract entities from claim (LLM or regex)
    2. Resolve to Wikidata items (LLM disambiguation)
    3. Retrieve bidirectional facts (outgoing + incoming)
    4. Beam search expand with LLM-guided pruning
    """
    entities = extract_entities(claim, llm_fn)
    evidence = {"entities": [], "facts": [], "expanded": []}

    for name in entities[:5]:
        resolved = resolve_entity(name, claim=claim, llm_fn=llm_fn)
        if resolved:
            evidence["entities"].append(resolved)
            # Bidirectional fact retrieval
            facts = get_entity_facts_bidirectional(resolved["id"])
            evidence["facts"].extend([
                {**f, "entity": resolved["label"], "entity_id": resolved["id"]}
                for f in facts
            ])

    # Beam search expansion from top entity
    if evidence["entities"]:
        top = evidence["entities"][0]
        expanded = beam_search_expand(
            top["id"], top["label"], claim, 
            llm_fn=llm_fn, beam_width=beam_width, max_hops=max_hops
        )
        evidence["expanded"] = expanded

        # Get facts for expanded entities too (1st hop only to avoid explosion)
        for exp in expanded[:3]:
            if exp.get("hop") == 1:
                target_name = exp["related_entity"]
                target = resolve_entity(target_name, claim=claim)
                if target:
                    extra_facts = get_entity_facts_bidirectional(target["id"], max_facts=10)
                    evidence["facts"].extend([
                        {**f, "entity": target["label"], "entity_id": target["id"]}
                        for f in extra_facts
                    ])

    return evidence
