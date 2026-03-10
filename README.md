# FactCheck 🔍

**Open-source hybrid fact verification using Knowledge Graphs + Web Search**

Inspired by [WKGFC](https://arxiv.org/abs/2603.00267) — a multi-agent evidence retrieval system for fact-checking. FactCheck implements the core hybrid retrieval pattern: structured knowledge graph facts (Wikidata) anchored with web search evidence, orchestrated by an LLM agent that decides what to look up.

## How It Works

```
Claim → Extract Entities → Wikidata SPARQL → Assess Sufficiency
                                                     │
                                          ┌──────────┴──────────┐
                                          │                     │
                                     Sufficient            Insufficient
                                          │                     │
                                          ▼                     ▼
                                       Verdict          Web Search → Verdict
```

1. **Entity Extraction** — LLM or regex-based NER pulls key entities from the claim
2. **Knowledge Graph Retrieval** — Entities are resolved against Wikidata, facts retrieved via SPARQL, and the graph expanded by following relevant relations
3. **Sufficiency Assessment** — LLM decides if KG evidence is enough or if web search is needed
4. **Web Search** — Brave Search API (or DuckDuckGo fallback) retrieves supporting/contradicting evidence
5. **Verdict** — LLM synthesises all evidence into SUPPORTED / REFUTED / NOT ENOUGH EVIDENCE with citations and confidence rating

## Quick Start

```bash
# Install
pip install -e .

# Verify a claim (uses local Ollama by default)
factcheck verify "The Eiffel Tower is located in Berlin"

# Use a specific model
factcheck verify "Einstein was born in Germany" --model qwen2.5:14b

# Use OpenAI instead of Ollama
factcheck verify "The speed of light is 300,000 km/s" --provider openai

# JSON output
factcheck verify "Marie Curie won two Nobel Prizes" --json-output

# Batch verification
factcheck batch claims.txt --output results.json

# Verbose mode (shows step-by-step)
factcheck verify "Shakespeare wrote Hamlet" -v
```

## Example Output

```
🔎 Checking: The Eiffel Tower is located in Berlin

🔍 Step 1: Searching Knowledge Graph...
   Found 15 facts, 2 entities
🌐 Step 2: KG insufficient, searching web...
   Found 5 web results, 2 page extracts
⚖️  Step 3: Delivering verdict...

╭─────────── ⚖️  Verdict ────────────╮
│ REFUTED  🟢 Confidence: HIGH       │
╰─────────────────────────────────────╯

┌──────── Evidence Gathered ────────┐
│ Source        │ Items             │
├───────────────┼───────────────────┤
│ KG Entities   │ 2                 │
│ KG Facts      │ 15                │
│ Web Results   │ 5                 │
│ Page Extracts │ 2                 │
└───────────────┴───────────────────┘
```

## Providers

| Provider | Setup | Cost |
|----------|-------|------|
| **Ollama** (default) | `ollama pull qwen2.5:14b` | Free (local) |
| **OpenAI** | Set `OPENAI_API_KEY` | ~$0.01/claim |
| **Any OpenAI-compatible** | Custom base URL | Varies |

## Web Search

By default, uses DuckDuckGo instant answers (no API key needed). For better results, set a Brave Search API key:

```bash
export BRAVE_API_KEY=your_key_here
```

## Architecture

The design is deliberately simple — 4 Python files, no ML training, no complex dependencies:

- **`kg.py`** — Wikidata SPARQL retrieval with entity resolution and graph expansion
- **`web.py`** — Web search via Brave/DuckDuckGo with page extraction
- **`agent.py`** — Agentic loop: KG → assess → web (if needed) → verdict
- **`cli.py`** — Rich CLI with tables, panels, and batch mode

### Key Differences from WKGFC

| WKGFC (Paper) | FactCheck (This) |
|---------------|-----------------|
| Full MDP formulation | Simple 3-step loop |
| TextGrad prompt optimization | Zero-shot prompting |
| SpaCy NER | LLM-based or regex NER |
| Custom KG construction | Wikidata (free, no setup) |
| GPT-4o only | Any LLM (Ollama, OpenAI, etc.) |
| Research prototype | Installable CLI tool |

## Use Cases

- **Content verification** — Fact-check AI-generated articles before publishing
- **Research** — Quick claim verification with structured evidence
- **Education** — Teach critical thinking with transparent reasoning chains
- **Journalism** — First-pass verification with citation trails

## License

MIT

## Credits

Inspired by:
- [WKGFC](https://arxiv.org/abs/2603.00267) — Gong et al., "Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking"
- [Wikidata](https://www.wikidata.org/) — Free structured knowledge base
- [Brave Search](https://brave.com/search/api/) — Privacy-respecting search API
