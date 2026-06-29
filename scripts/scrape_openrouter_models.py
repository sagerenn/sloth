#!/usr/bin/env python3
"""Scrape the top-N most-popular OpenRouter models into the model_catalog dir.

The catalog format (see src/model_catalog.rs) is one YAML file per model:

    models:        # or a bare single-model map
      - id: deepseek/deepseek-v4-flash
        provider: deepseek
        context_window: 1048576
        max_output: 65536
        modalities:
          input: ["text"]
          output: ["text"]
        pricing:
          prompt_per_1m: 0.09
          completion_per_1m: 0.18
        scores:
          average: 1300.0

Popularity ordering comes from OpenRouter's frontend endpoint
(`/api/frontend/v1/models/find?order=most-popular`); pricing, context,
max_output, modalities, and benchmarks are enriched from the public
`/api/v1/models` endpoint. Pricing on OpenRouter is USD-per-token, so we
multiply by 1_000_000 to get the catalog's per-1M-token format.

Usage: python3 scripts/scrape_openrouter_models.py [N]
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import yaml

CATALOG_DIR = Path(__file__).resolve().parent.parent / "model_catalog"
DEFAULT_N = 30

POPULARITY_URL = "https://openrouter.ai/api/frontend/v1/models/find?order=most-popular"
MODELS_URL = "https://openrouter.ai/api/v1/models"
UA = "sloth-agent/model-catalog-scraper"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def to_per_1m(price_per_token: str | None) -> float:
    """OpenRouter pricing is USD/token; the catalog wants USD / 1M tokens."""
    if price_per_token in (None, ""):
        return 0.0
    try:
        return round(float(price_per_token) * 1_000_000, 6)
    except ValueError:
        return 0.0


def collect_scores(benchmarks: dict | None) -> dict[str, float]:
    """Flatten OpenRouter's benchmark blobs into a full scores map.

    OpenRouter exposes two benchmark arenas:

    * ``design_arena`` — a list of ``{arena, category, elo, win_rate, rank}``
      entries. We emit one key per (arena, category) for ``elo`` and
      ``win_rate``, e.g. ``elo:agents:fullstack`` and ``win_rate:models:3d``.
    * ``artificial_analysis`` — a flat map of named indices, e.g.
      ``intelligence_index`` / ``coding_index`` / ``agentic_index``. We emit
      each as-is.

    We also set ``average`` to the mean elo across all design_arena entries —
    this is the catalog's default ranking score (higher = better) — so models
    that only carry the artificial_analysis indices still get a sensible tier.
    """
    scores: dict[str, float] = {}
    if not benchmarks:
        return scores

    elos: list[float] = []
    da = benchmarks.get("design_arena")
    if isinstance(da, list):
        for e in da:
            if not isinstance(e, dict):
                continue
            arena = e.get("arena", "arena")
            cat = e.get("category", "cat")
            if isinstance(e.get("elo"), (int, float)):
                elo = float(e["elo"])
                scores[f"elo:{arena}:{cat}"] = round(elo, 1)
                elos.append(elo)
            if isinstance(e.get("win_rate"), (int, float)):
                scores[f"win_rate:{arena}:{cat}"] = round(float(e["win_rate"]), 1)
            if isinstance(e.get("rank"), (int, float)):
                scores[f"rank:{arena}:{cat}"] = float(e["rank"])

    aa = benchmarks.get("artificial_analysis")
    if isinstance(aa, dict):
        for k, v in aa.items():
            if isinstance(v, (int, float)):
                scores[k] = round(float(v), 2)

    if elos:
        scores["average"] = round(sum(elos) / len(elos), 1)
    return scores


def main(n: int = DEFAULT_N) -> None:
    popular = fetch_json(POPULARITY_URL)
    pop_models = popular["data"]["models"]
    popular_slugs = [m["slug"] for m in pop_models if not m.get("hidden", False)]

    models_by_id = {m["id"]: m for m in fetch_json(MODELS_URL)["data"]}

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for slug in popular_slugs:
        if written >= n:
            break
        m = models_by_id.get(slug)
        if m is None:
            print(f"skip {slug}: not in /api/v1/models", file=sys.stderr)
            continue

        arch = m.get("architecture") or {}
        pricing = m.get("pricing") or {}
        top = m.get("top_provider") or {}
        provider = slug.split("/", 1)[0] if "/" in slug else (m.get("name") or slug)

        entry = {
            "id": m["id"],
            "name": m.get("name"),
            "provider": provider,
            "context_window": m.get("context_length", 0),
            "max_output": top.get("max_completion_tokens"),
            "modalities": {
                "input": arch.get("input_modalities") or [],
                "output": arch.get("output_modalities") or [],
                "wire": arch.get("modality"),
            },
            "pricing": {
                "prompt_per_1m": to_per_1m(pricing.get("prompt")),
                "completion_per_1m": to_per_1m(pricing.get("completion")),
            },
            "scores": collect_scores(m.get("benchmarks")),
        }

        # Strip None values + empty containers for a cleaner file.
        entry = {k: v for k, v in entry.items() if v is not None}
        if entry.get("max_output") is None:
            entry.pop("max_output", None)
        if not entry.get("scores"):
            entry.pop("scores", None)

        out = CATALOG_DIR / f"{slug.replace('/', '__')}.yaml"
        header = "# Scraped from OpenRouter (most-popular). See scripts/scrape_openrouter_models.py\n"
        out.write_text(header + yaml.dump({"models": [entry]}, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written += 1
        print(f"[{written:>2}/{n}] {slug}  ->  {out.name}")

    print(f"\nWrote {written} models to {CATALOG_DIR}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    main(n)
