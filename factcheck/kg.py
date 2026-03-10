"""Knowledge Graph retrieval via Wikidata SPARQL endpoint."""

import re
import json
import requests
from SPARQLWrapper import SPARQLWrapper, JSON


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
sparql.setReturnFormat(JSON)
sparql.addCustomHttpHeader("User-Agent", "FactCheck/0.1 (https://github.com/jhammant/factcheck)")


def extract_entities(text: str, llm_fn=None) -> list[str]:
    """Extract entity names from a claim. Uses LLM if available, else simple NER."""
    if llm_fn:
        prompt = (
            f"Extract the key named entities (people, places, organisations, events, things) "
            f"from this claim. Return ONLY a JSON array of strings, nothing else.\n\n"
            f"Claim: \"{text}\"\n\nEntities:"
        )
        resp = llm_fn(prompt)
        try:
            match = re.search(r'\[.*?\]', resp, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                if entities:
                    return entities
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: capitalised word sequences (simple NER)
    # First try multi-word entities (e.g. "Eiffel Tower", "Marie Curie")
    entities = re.findall(r'(?:[A-Z][a-z]+(?:\s+(?:the\s+)?[A-Z][a-z]+)*)', text)
    # Also try "The X" patterns (e.g. "The Eiffel Tower")
    the_entities = re.findall(r'[Tt]he\s+((?:[A-Z][a-z]+\s*)+)', text)
    entities.extend([e.strip() for e in the_entities])
    
    # Filter common words
    stop = {"This", "That", "These", "Those", "There", "When", "Where",
            "What", "Which", "How", "Who", "Its", "His", "Her", "Did", "Was",
            "Were", "Has", "Have", "Had", "Are", "Been", "Being", "Does", "Do",
            "Located", "Born", "Died", "Known", "Won", "Made", "Called", "Also"}
    
    # Deduplicate while preserving order
    seen = set()
    result = []
    for e in entities:
        if e not in stop and len(e) > 1 and e.lower() not in seen:
            seen.add(e.lower())
            result.append(e)
    return result


def resolve_entity(name: str) -> dict | None:
    """Resolve an entity name to a Wikidata item via search API.
    Prefers items with descriptions suggesting real-world entities over artworks etc."""
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "limit": 7,
        "format": "json",
    }
    try:
        resp = requests.get(WIKIDATA_API, params=params, timeout=10,
                          headers={"User-Agent": "FactCheck/0.1"})
        data = resp.json()
        results = data.get("search", [])
        if not results:
            return None

        # Prefer real-world entities: deprioritise paintings, songs, films, episodes etc.
        artwork_terms = {"painting", "song", "album", "film", "episode", "novel",
                        "sculpture", "photograph", "artwork", "book by", "poem"}

        # Score each result
        best = None
        best_score = -1
        for r in results:
            desc = (r.get("description") or "").lower()
            score = 0
            # Boost for geographic/person/org descriptions
            if any(t in desc for t in ["city", "country", "capital", "state", "continent",
                                        "person", "politician", "scientist", "born",
                                        "company", "organisation", "organization",
                                        "tower", "building", "structure", "landmark",
                                        "river", "mountain", "island", "university"]):
                score += 10
            # Penalise artworks
            if any(t in desc for t in artwork_terms):
                score -= 5
            # Boost if description is non-empty (better documented entities)
            if desc:
                score += 1
            # Boost shorter QIDs (more notable entities tend to have lower numbers)
            qid = r.get("id", "Q999999999")
            try:
                qnum = int(qid[1:])
                if qnum < 1000:
                    score += 5
                elif qnum < 10000:
                    score += 3
                elif qnum < 100000:
                    score += 1
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


def get_entity_facts(entity_id: str, max_facts: int = 25) -> list[dict]:
    """Retrieve key facts about an entity from Wikidata with human-readable labels."""
    query = f"""
    SELECT ?propertyLabel ?valueLabel WHERE {{
      wd:{entity_id} ?directProp ?value .
      ?property wikibase:directClaim ?directProp .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT {max_facts}
    """
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        facts = []
        seen = set()
        for r in results["results"]["bindings"]:
            prop = r.get("propertyLabel", {}).get("value", "")
            val = r.get("valueLabel", {}).get("value", "")
            # Skip if property or value is a raw URI
            if prop.startswith("http") or val.startswith("http"):
                continue
            # Skip if value is just a QID or P-number
            if re.match(r'^[QP]\d+$', val) or re.match(r'^[QP]\d+$', prop):
                continue
            # Skip technical/ID properties
            skip_props = {"Freebase ID", "MeSH descriptor ID", "MeSH tree code",
                         "NL CR AUT ID", "NUTS code", "GND ID", "VIAF ID",
                         "Library of Congress authority ID", "BnF ID",
                         "IMDb ID", "GeoNames ID", "OpenStreetMap relation ID",
                         "ISO 3166-2 code", "local dialing code", "postal code",
                         "coordinate location", "Commons category", "topic's main category",
                         "page banner", "image", "flag image", "coat of arms image",
                         "locator map image", "pronunciation audio", "spoken text audio"}
            if any(sp.lower() in prop.lower() for sp in skip_props):
                continue
            # Skip values that look like IDs, coordinates, or timestamps
            if re.match(r'^[\d/\-T:.]+Z?$', val):
                continue
            if val.startswith("Point("):
                continue
            key = f"{prop}:{val}"
            if key not in seen:
                seen.add(key)
                facts.append({"property": prop, "value": val})
        return facts
    except Exception as e:
        return [{"property": "error", "value": str(e)}]


def expand_entity(entity_id: str, claim: str, llm_fn=None) -> list[dict]:
    """Expand knowledge graph by following relations relevant to the claim."""
    query = f"""
    SELECT ?relatedLabel ?propLabel ?valueLabel WHERE {{
      wd:{entity_id} ?p1 ?related .
      ?related ?p2 ?value .
      ?prop1 wikibase:directClaim ?p1 .
      ?prop2 wikibase:directClaim ?p2 .
      FILTER(!STRSTARTS(STR(?value), "http://www.wikidata.org"))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    LIMIT 30
    """
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        expanded = []
        seen = set()
        for r in results["results"]["bindings"]:
            related = r.get("relatedLabel", {}).get("value", "")
            prop = r.get("propLabel", {}).get("value", "")
            val = r.get("valueLabel", {}).get("value", "")
            # Skip raw URIs and QIDs
            if (related.startswith("http") or prop.startswith("http") or
                val.startswith("http") or re.match(r'^Q\d+$', val)):
                continue
            key = f"{related}:{prop}:{val}"
            if key not in seen:
                seen.add(key)
                expanded.append({
                    "related_entity": related,
                    "property": prop,
                    "value": val,
                })
        return expanded
    except Exception:
        return []


def retrieve_kg_evidence(claim: str, llm_fn=None) -> dict:
    """Full KG retrieval pipeline: extract entities → resolve → get facts → expand."""
    entities = extract_entities(claim, llm_fn)
    evidence = {"entities": [], "facts": [], "expanded": []}

    for name in entities[:5]:  # Cap at 5 entities
        resolved = resolve_entity(name)
        if resolved:
            evidence["entities"].append(resolved)
            facts = get_entity_facts(resolved["id"])
            evidence["facts"].extend([
                {**f, "entity": resolved["label"], "entity_id": resolved["id"]}
                for f in facts if f.get("property") != "error"
            ])

    # Expand top entity if we have one
    if evidence["entities"]:
        top_entity = evidence["entities"][0]
        expanded = expand_entity(top_entity["id"], claim, llm_fn)
        evidence["expanded"] = expanded[:15]

    return evidence
