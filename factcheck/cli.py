"""CLI interface for FactCheck."""

import json
import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from .agent import verify_claim


console = Console()


VERDICT_COLORS = {
    "SUPPORTED": "green",
    "REFUTED": "red",
    "NOT ENOUGH EVIDENCE": "yellow",
}

CONFIDENCE_EMOJI = {
    "HIGH": "🟢",
    "MEDIUM": "🟡",
    "LOW": "🔴",
}


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """FactCheck — Hybrid fact verification using Knowledge Graphs + Web Search."""
    pass


@cli.command()
@click.argument("claim")
@click.option("--provider", "-p", default="ollama", help="LLM provider: ollama, openai")
@click.option("--model", "-m", default=None, help="Model name (default: qwen2.5:14b for ollama)")
@click.option("--brave-key", envvar="BRAVE_API_KEY", default=None, help="Brave Search API key")
@click.option("--verbose", "-v", is_flag=True, help="Show step-by-step progress")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def verify(claim, provider, model, brave_key, verbose, json_output):
    """Verify a claim against Knowledge Graph and web evidence."""
    if not json_output:
        console.print(f"\n🔎 Checking: [italic]{claim}[/italic]\n")

    result = verify_claim(
        claim=claim,
        provider=provider,
        model=model,
        brave_api_key=brave_key,
        verbose=verbose,
    )

    if json_output:
        print(json.dumps(result, indent=2))
        return

    # Verdict panel
    color = VERDICT_COLORS.get(result["verdict"], "white")
    emoji = CONFIDENCE_EMOJI.get(result["confidence"], "⚪")

    console.print(Panel(
        f"[bold {color}]{result['verdict']}[/bold {color}]  {emoji} Confidence: {result['confidence']}",
        title="⚖️  Verdict",
        border_style=color,
    ))

    # Evidence summary
    table = Table(title="Evidence Gathered", show_header=True)
    table.add_column("Source", style="cyan")
    table.add_column("Items", style="green")
    table.add_row("KG Entities", str(result["evidence"]["kg"]["entities"]))
    table.add_row("KG Facts", str(result["evidence"]["kg"]["facts"]))
    table.add_row("Web Results", str(result["evidence"]["web"]["results"]))
    table.add_row("Page Extracts", str(result["evidence"]["web"]["pages"]))
    console.print(table)

    # Explanation
    console.print("\n[bold]Explanation:[/bold]")
    console.print(Panel(result["explanation"], border_style="dim"))


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--provider", "-p", default="ollama", help="LLM provider")
@click.option("--model", "-m", default=None, help="Model name")
@click.option("--brave-key", envvar="BRAVE_API_KEY", default=None)
@click.option("--output", "-o", default=None, help="Output JSON file")
def batch(input_file, provider, model, brave_key, output):
    """Verify multiple claims from a file (one claim per line or JSON array)."""
    # Load claims
    with open(input_file) as f:
        content = f.read().strip()
        try:
            claims = json.loads(content)
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                pass
            elif isinstance(claims, list) and all(isinstance(c, dict) for c in claims):
                claims = [c.get("claim", c.get("text", "")) for c in claims]
            else:
                claims = content.splitlines()
        except json.JSONDecodeError:
            claims = [line.strip() for line in content.splitlines() if line.strip()]

    results = []
    for i, claim in enumerate(claims, 1):
        console.print(f"\n[bold]Claim {i}/{len(claims)}:[/bold] {claim}")
        result = verify_claim(
            claim=claim, provider=provider, model=model,
            brave_api_key=brave_key, verbose=False,
        )
        results.append(result)
        color = VERDICT_COLORS.get(result["verdict"], "white")
        console.print(f"  → [{color}]{result['verdict']}[/{color}] ({result['confidence']})")

    # Summary table
    console.print("\n")
    table = Table(title=f"Results: {len(results)} claims")
    table.add_column("#", style="dim")
    table.add_column("Claim", max_width=60)
    table.add_column("Verdict")
    table.add_column("Confidence")

    for i, r in enumerate(results, 1):
        color = VERDICT_COLORS.get(r["verdict"], "white")
        table.add_row(str(i), r["claim"][:60], f"[{color}]{r['verdict']}[/{color}]", r["confidence"])

    console.print(table)

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n📄 Results saved to {output}")


def main():
    cli()


if __name__ == "__main__":
    main()
