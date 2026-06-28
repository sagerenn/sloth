#!/usr/bin/env python3
# Copyright AGNTCY Contributors (https://github.com/agntcy)
# SPDX-License-Identifier: Apache-2.0
"""
ITK results post-processor for a2a-rs nightly and CI runs.

Usage
-----
CI mode  — reads raw JSON from stdin, prints per-test status, exits 1 on failure:
    echo "$RESPONSE" | python3 process_results.py ci

Nightly mode — reads raw_results.json, fetches rolling history, appends a new
run entry and writes the updated history to the output file:
    python3 process_results.py nightly \\
        --scenarios  scenarios_full.json \\
        --output     itk_rust.json \\
        --history-url https://github.com/a2aproject/a2a-rs/releases/download/nightly-metrics/itk_rust.json
"""

import argparse
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

RAW_RESULTS_FILE = "raw_results.json"
HISTORY_LIMIT = 50


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _passed(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("passed", False))
    return False


# ---------------------------------------------------------------------------
# CI mode
# ---------------------------------------------------------------------------

def _check_itk_error(data: dict | list | None) -> int | None:
    """Return 1 and print error if ITK service returned an error response, else None."""
    if not isinstance(data, dict):
        print(f"ERROR: ITK response is not a JSON object. Type: {type(data).__name__}", file=sys.stderr)
        return 1
    if "detail" in data:
        print(f"ERROR: ITK service returned an error: {data['detail']}", file=sys.stderr)
        return 1
    if "results" not in data:
        print(f"ERROR: ITK response missing 'results' field. Response keys: {list(data.keys())}", file=sys.stderr)
        return 1
    return None


def cmd_ci(args) -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"ERROR: could not parse response JSON: {exc}", file=sys.stderr)
        return 1

    if (err := _check_itk_error(data)) is not None:
        return err

    results = data.get("results", {})
    all_passed = data.get("all_passed", False)

    print("-" * 56)
    print("ITK TEST RESULTS:")
    print("-" * 56)
    for name, value in results.items():
        status = "PASSED" if _passed(value) else "FAILED"
        print(f"{name}: {status}")
    print("-" * 56)
    print(f"OVERALL STATUS: {'PASSED' if all_passed else 'FAILED'}")

    return 0 if all_passed else 1


# ---------------------------------------------------------------------------
# Nightly mode
# ---------------------------------------------------------------------------

def _fetch_history(url: str) -> list:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "a2a-rs/itk"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not fetch history ({exc}); starting fresh", file=sys.stderr)
        return []


def _load_json(path: str) -> dict | list:
    with open(path) as fh:
        return json.load(fh)


def _compile_scenarios(raw_results: dict, base_tests: list) -> list:
    index = {t["name"]: t for t in base_tests}
    compiled = []
    for name, value in raw_results.items():
        parent = name.split("-sub-")[0]
        base = index.get(parent)
        if base is None:
            print(f"WARNING: no base scenario for result key '{name}'; skipping", file=sys.stderr)
            continue

        is_dict = isinstance(value, dict)
        record = {
            "name": name,
            "sdks": value.get("sdks", base["sdks"]) if is_dict else base["sdks"],
            "edges": value.get("edges", base.get("edges")) if is_dict else base.get("edges"),
            "protocols": base.get("protocols"),
            "behavior": base.get("behavior"),
            "traversal": base.get("traversal", "euler"),
            "passed": _passed(value),
        }
        for opt in ("streaming", "build_subtests"):
            if opt in base:
                record[opt] = base[opt]
        compiled.append(record)
    return compiled


def cmd_nightly(args) -> int:
    raw = _load_json(RAW_RESULTS_FILE)
    if (err := _check_itk_error(raw)) is not None:
        return err

    scenarios_doc = _load_json(args.scenarios)
    history = _fetch_history(args.history_url)

    compiled = _compile_scenarios(raw.get("results", {}), scenarios_doc["tests"])

    new_run = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit_sha": os.environ.get("GITHUB_SHA", "local-dev"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", "0"),
        "all_passed": raw.get("all_passed", False),
        "scenarios": compiled,
    }

    history.append(new_run)
    if len(history) > HISTORY_LIMIT:
        history = history[-HISTORY_LIMIT:]

    with open(args.output, "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"Written {len(history)} history entries to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ci", help="Print CI test results (reads JSON from stdin)")

    nightly = sub.add_parser("nightly", help="Build and persist nightly history")
    nightly.add_argument("--scenarios",   required=True, help="Path to scenarios JSON file")
    nightly.add_argument("--output",      required=True, help="Output history JSON file")
    nightly.add_argument("--history-url", required=True, dest="history_url",
                         help="URL of the existing rolling history release asset")

    args = parser.parse_args()
    return cmd_ci(args) if args.command == "ci" else cmd_nightly(args)


if __name__ == "__main__":
    sys.exit(main())
