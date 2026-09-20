"""FactCheck Web Demo — FastAPI backend serving a single-page verification UI."""

import asyncio
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="FactCheck", description="Hybrid fact verification with Knowledge Graphs + Web Search")

# Import verify_claim
from factcheck.agent import verify_claim


@app.get("/", response_class=HTMLResponse)
async def index():
    return FRONTEND_HTML


@app.post("/api/verify")
async def verify(request: Request):
    body = await request.json()
    claim = body.get("claim", "").strip()
    provider = body.get("provider", "ollama")
    model = body.get("model", "llama3.2:3b")
    mode = body.get("mode", "fast")

    if not claim:
        return JSONResponse({"error": "No claim provided"}, status_code=400)

    try:
        result = await asyncio.to_thread(
            verify_claim,
            claim=claim,
            provider=provider,
            model=model,
            mode=mode,
            max_hops=2,
            beam_width=5,
            verbose=False,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FactCheck — Verify Any Claim</title>
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
    --text: #e6edf3; --muted: #8b949e; --border: #30363d;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --blue: #58a6ff; --purple: #bc8cff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

  .container { max-width: 720px; margin: 0 auto; padding: 2rem 1.5rem; }

  h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
  h1 span { color: var(--purple); }
  .subtitle { color: var(--muted); margin-bottom: 2rem; font-size: 0.95rem; }

  .input-group { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  .input-group label { display: block; color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }

  textarea { width: 100%; background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 0.75rem 1rem; font-size: 1rem; font-family: inherit; resize: vertical; min-height: 80px; }
  textarea:focus { outline: none; border-color: var(--purple); }

  .options { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
  .options select { background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 0.5rem 0.75rem; font-size: 0.9rem; flex: 1; min-width: 120px; }
  .options select:focus { outline: none; border-color: var(--purple); }

  .btn { background: var(--purple); color: #fff; border: none; border-radius: 8px; padding: 0.75rem 2rem; font-size: 1rem; font-weight: 600; cursor: pointer; width: 100%; margin-top: 1rem; transition: opacity 0.2s; }
  .btn:hover { opacity: 0.9; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .result { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; display: none; }
  .result.visible { display: block; }

  .verdict-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .verdict-badge { padding: 0.4rem 1rem; border-radius: 20px; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .verdict-SUPPORTED { background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid rgba(63,185,80,0.3); }
  .verdict-REFUTED { background: rgba(248,81,73,0.15); color: var(--red); border: 1px solid rgba(248,81,73,0.3); }
  .verdict-NEI { background: rgba(210,153,34,0.15); color: var(--yellow); border: 1px solid rgba(210,153,34,0.3); }
  .confidence { color: var(--muted); font-size: 0.85rem; }

  .explanation { color: var(--text); line-height: 1.6; margin-bottom: 1rem; font-size: 0.95rem; }

  .evidence-section h3 { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
  .evidence-item { background: var(--bg3); border-radius: 6px; padding: 0.5rem 0.75rem; margin-bottom: 0.4rem; font-size: 0.85rem; font-family: 'SF Mono', 'Fira Code', monospace; color: var(--muted); word-break: break-word; }
  .evidence-item .arrow { color: var(--blue); }

  .spinner { display: none; text-align: center; padding: 2rem; color: var(--muted); }
  .spinner.visible { display: block; }
  .spinner .dots { font-size: 1.5rem; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }

  .error { background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.3); border-radius: 8px; padding: 1rem; color: var(--red); margin-top: 1rem; display: none; }
  .error.visible { display: block; }

  .examples { margin-top: 2rem; }
  .examples h3 { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem; }
  .example-chip { display: inline-block; background: var(--bg3); border: 1px solid var(--border); border-radius: 20px; padding: 0.4rem 0.85rem; font-size: 0.8rem; color: var(--muted); cursor: pointer; margin: 0.25rem; transition: all 0.2s; }
  .example-chip:hover { border-color: var(--purple); color: var(--text); }

  .footer { text-align: center; margin-top: 3rem; color: var(--muted); font-size: 0.8rem; }
  .footer a { color: var(--purple); text-decoration: none; }

  @media (max-width: 600px) {
    .container { padding: 1.5rem 1rem; }
    h1 { font-size: 1.5rem; }
    .options { flex-direction: column; }
  }
</style>
</head>
<body>
<div class="container">
  <h1>🔍 Fact<span>Check</span></h1>
  <p class="subtitle">Verify any claim using Knowledge Graphs + Web Search + LLMs</p>

  <div class="input-group">
    <label>Enter a claim to verify</label>
    <textarea id="claim" placeholder="e.g. The capital of Australia is Sydney"></textarea>
    <div class="options">
      <select id="provider">
        <option value="ollama">Ollama (Local)</option>
        <option value="openai">OpenAI</option>
        <option value="gemini">Gemini</option>
      </select>
      <select id="model">
        <option value="llama3.2:3b">llama3.2:3b</option>
        <option value="qwen2.5:14b">qwen2.5:14b</option>
        <option value="gpt-4o-mini">gpt-4o-mini</option>
        <option value="gemini-2.5-flash">gemini-2.5-flash</option>
      </select>
      <select id="mode">
        <option value="fast">Fast</option>
        <option value="deep">Deep</option>
      </select>
    </div>
    <button class="btn" id="verifyBtn" onclick="verify()">Verify Claim</button>
  </div>

  <div class="spinner" id="spinner">
    <div class="dots">⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏</div>
    <p>Checking Wikidata Knowledge Graph + Web sources…</p>
    <p style="font-size:0.8rem; margin-top:0.5rem;">This typically takes 30–60 seconds</p>
  </div>

  <div class="error" id="error"></div>

  <div class="result" id="result">
    <div class="verdict-row">
      <span class="verdict-badge" id="verdict"></span>
      <span class="confidence" id="confidence"></span>
    </div>
    <div class="explanation" id="explanation"></div>
    <div class="evidence-section" id="evidenceSection">
      <h3>Evidence</h3>
      <div id="evidenceList"></div>
    </div>
  </div>

  <div class="examples">
    <h3>Try these</h3>
    <span class="example-chip" onclick="tryExample(this)">The capital of Australia is Sydney</span>
    <span class="example-chip" onclick="tryExample(this)">Marie Curie won two Nobel Prizes</span>
    <span class="example-chip" onclick="tryExample(this)">Mount Everest is in Africa</span>
    <span class="example-chip" onclick="tryExample(this)">The Berlin Wall fell in 1991</span>
    <span class="example-chip" onclick="tryExample(this)">Shakespeare wrote War and Peace</span>
    <span class="example-chip" onclick="tryExample(this)">Bitcoin was invented by Satoshi Nakamoto</span>
  </div>

  <div class="footer">
    <p>Powered by <a href="https://github.com/jhammant/factcheck">FactCheck</a> — Wikidata SPARQL + Web Search + LLM reasoning</p>
    <p style="margin-top:0.25rem;">MIT License · pip install factcheck-kg</p>
  </div>
</div>

<script>
function tryExample(el) {
  document.getElementById('claim').value = el.textContent;
}

async function verify() {
  const claim = document.getElementById('claim').value.trim();
  if (!claim) return;

  const btn = document.getElementById('verifyBtn');
  const spinner = document.getElementById('spinner');
  const result = document.getElementById('result');
  const error = document.getElementById('error');

  btn.disabled = true;
  btn.textContent = 'Verifying…';
  spinner.classList.add('visible');
  result.classList.remove('visible');
  error.classList.remove('visible');

  try {
    const base = window.location.pathname.replace(/\/$/, '');
    const resp = await fetch(base + '/api/verify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        claim,
        provider: document.getElementById('provider').value,
        model: document.getElementById('model').value,
        mode: document.getElementById('mode').value,
      })
    });
    const data = await resp.json();

    if (data.error) {
      error.textContent = data.error;
      error.classList.add('visible');
      return;
    }

    // Verdict badge
    const verdict = data.verdict || 'UNKNOWN';
    const vEl = document.getElementById('verdict');
    vEl.textContent = verdict;
    vEl.className = 'verdict-badge';
    if (verdict === 'SUPPORTED') vEl.classList.add('verdict-SUPPORTED');
    else if (verdict === 'REFUTED') vEl.classList.add('verdict-REFUTED');
    else vEl.classList.add('verdict-NEI');

    // Confidence
    document.getElementById('confidence').textContent =
      data.confidence ? `Confidence: ${data.confidence}` : '';

    // Explanation
    document.getElementById('explanation').textContent = data.explanation || '';

    // Evidence
    const eList = document.getElementById('evidenceList');
    eList.innerHTML = '';
    const ev = data.evidence || {};
    const kgTrips = ev.kg_triples || ev.kg_triplets || [];
    const webSnips = ev.web_snippets || [];

    if (kgTrips.length > 0) {
      kgTrips.forEach(t => {
        const div = document.createElement('div');
        div.className = 'evidence-item';
        div.innerHTML = `<span class="arrow">→</span> ${escHtml(typeof t === 'string' ? t : JSON.stringify(t))}`;
        eList.appendChild(div);
      });
    }
    if (webSnips.length > 0) {
      webSnips.slice(0, 5).forEach(s => {
        const div = document.createElement('div');
        div.className = 'evidence-item';
        const txt = typeof s === 'string' ? s : (s.text || s.snippet || JSON.stringify(s));
        div.innerHTML = `<span class="arrow">🌐</span> ${escHtml(txt).substring(0, 300)}`;
        eList.appendChild(div);
      });
    }

    if (kgTrips.length === 0 && webSnips.length === 0) {
      document.getElementById('evidenceSection').style.display = 'none';
    } else {
      document.getElementById('evidenceSection').style.display = 'block';
    }

    result.classList.add('visible');
  } catch (e) {
    error.textContent = 'Connection error: ' + e.message;
    error.classList.add('visible');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Verify Claim';
    spinner.classList.remove('visible');
  }
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

document.getElementById('claim').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); verify(); }
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
