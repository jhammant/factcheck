# 🔍 FactCheck

Open-source hybrid fact verification using **Knowledge Graphs** + **Web Search** + **LLM reasoning**.

Inspired by [WKGFC](https://arxiv.org/abs/2603.00267) (Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking, SIGIR 2026).

## How It Works

```
Claim → Entity Extraction → Wikidata KG (bidirectional SPARQL)
                                ↓
                         Beam Search Expansion (multi-hop)
                                ↓
                    Evidence Sufficient? ─── Yes → LLM Verdict
                                │
                               No
                                ↓
                    Web Search (Brave/Wikipedia/DDG)
                                ↓
                         LLM Verdict + Citations
```

**Two modes:**

| Mode | LLM Calls | Speed | Best For |
|------|-----------|-------|----------|
| `fast` | 2 | ~30s (local 3B) | Small/slow models, quick checks |
| `deep` | 4-8 | ~2min+ | Capable models (7B+, GPT-4, Gemini) |

## Key Features

- **Wikidata SPARQL** — Structured facts from 100M+ items (free, no API key)
- **Bidirectional retrieval** — Both outgoing AND incoming relations (catches "X is capital of Y" from either direction)
- **Beam search expansion** — Multi-hop graph traversal with relevance pruning
- **Wikipedia + Brave + DDG** — Multiple web sources with automatic fallback
- **Coarse-to-fine web filtering** — LLM scores web passages for factual relevance (deep mode)
- **Web→KG triplet extraction** — Converts web text into structured facts (deep mode)
- **Multi-provider LLM** — Ollama (local), OpenAI, Google Gemini
- **CLI with batch mode** — Verify hundreds of claims from a file

## Install

```bash
pip install -e .
```

Or with dependencies:
```bash
pip install click rich requests SPARQLWrapper
```

## Quick Start

```bash
# Local model (Ollama)
factcheck verify "The Eiffel Tower is in Berlin" --model llama3.2:3b

# OpenAI
factcheck verify "Marie Curie won two Nobel Prizes" -p openai -m gpt-4o-mini

# Gemini
factcheck verify "Bitcoin was invented by Satoshi Nakamoto" -p gemini

# Deep mode (more thorough, needs bigger model)
factcheck verify "Tesla was founded by Elon Musk" --mode deep -m qwen2.5:14b

# Batch mode
factcheck batch claims.txt --output results.json

# Verbose (see the evidence gathering process)
factcheck verify "The Great Wall of China is visible from space" -v
```

## Architecture (vs WKGFC paper)

| WKGFC Paper | FactCheck Implementation |
|-------------|------------------------|
| Wikidata SPARQL with expand-and-prune beam search | ✅ Bidirectional SPARQL + beam search expansion |
| LLM-guided relation pruning at each hop | ✅ In deep mode; heuristic in fast mode |
| Web retrieval with coarse-to-fine LLM filtering | ✅ In deep mode; raw results in fast mode |
| Web→KG triplet alignment | ✅ In deep mode |
| MDP agent loop (expandKG/webSearch/verdict) | ✅ In deep mode; fixed pipeline in fast mode |
| Self-reflection + prompt optimization | ❌ Not implemented (needs training data) |
| SpaCy NER for entity extraction | LLM-based extraction (more flexible) |

## Example Output

```
🔎 Checking: The capital of Australia is Sydney

╭──────────────── ⚖️  Verdict ─────────────────╮
│ REFUTED  🟢 Confidence: HIGH                  │
╰───────────────────────────────────────────────╯

Evidence:
  → [Australia] capital: Canberra
  ← [Canberra] capital of: Australia
  → [Sydney] instance of: city, million city
  
Explanation: The claim states Sydney is the capital, but Wikidata 
clearly shows Canberra is the capital of Australia. Sydney is the 
largest city but not the capital.
```

## Accuracy

Tested on 21 claims with llama3.2:3b (tiny model on ARM):

| Category | Result |
|----------|--------|
| Clear facts (Marie Curie, Einstein) | ✅ HIGH confidence |
| Geographic (Everest in Africa) | ✅ REFUTED correctly |
| Temporal (Berlin Wall 1991) | ✅ REFUTED correctly |
| Nuanced (Tesla founding, Cleopatra) | ⚠️ Sometimes wrong with 3B model |

**Accuracy scales with model size.** GPT-4o-mini or Gemini Flash give significantly better results, especially on nuanced claims.

## Providers

| Provider | Setup | Cost |
|----------|-------|------|
| Ollama (local) | `ollama pull llama3.2:3b` | Free |
| OpenAI | `export OPENAI_API_KEY=sk-...` | ~$0.001/claim |
| Gemini | `export GEMINI_API_KEY=AI...` | Free tier available |

## API Usage

```python
from factcheck.agent import verify_claim

result = verify_claim(
    claim="The Earth is flat",
    provider="ollama",
    model="llama3.2:3b",
    mode="fast",        # or "deep" for thorough checking
    max_hops=2,         # KG expansion depth
    beam_width=5,       # beam search width
)

print(result["verdict"])      # "REFUTED"
print(result["confidence"])   # "HIGH"
print(result["explanation"])  # Detailed reasoning
print(result["evidence"])     # KG + web evidence counts
```

## License

MIT

## Citation

If you use this in research:

```bibtex
@software{factcheck2026,
  author = {Hammant, Jonathan},
  title = {FactCheck: Hybrid Fact Verification with Knowledge Graphs},
  year = {2026},
  url = {https://github.com/jhammant/factcheck}
}
```

Inspired by:
```bibtex
@inproceedings{wkgfc2026,
  title={Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking},
  author={Gong, Shuzhi and others},
  booktitle={SIGIR},
  year={2026}
}
```
