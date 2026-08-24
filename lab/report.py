"""Printing and persistence for evaluation reports, generic over any aggregate() shape."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def print_report(report: dict) -> None:
    print()
    print("=" * 78)
    print(f"  {report['method']}")
    print("=" * 78)
    rows = {name: split["overall"] for name, split in report["splits"].items()}
    print(pd.DataFrame(rows).T)
    print()
    print("Breakdown by group (solved rate), where available")
    print("-" * 78)
    for name, split in report["splits"].items():
        if not split["by_group"]:
            continue
        cells = "  ".join(f"{g}={m.get('solved')}" for g, m in sorted(split["by_group"].items(),
                                                                       key=str))
        print(f"{name:<22}{cells}")
    print("=" * 78)


def print_comparison(reports: list[dict], split: str) -> None:
    print()
    print("=" * 78)
    print(f"  METHOD COMPARISON -- split: {split}")
    print("=" * 78)
    rows = {}
    for report in reports:
        if split in report["splits"]:
            rows[report["method"]] = report["splits"][split]["overall"]
        else:
            rows[report["method"]] = {"n": "(missing)"}
    print(pd.DataFrame(rows).T)
    print("=" * 78)


def print_example(completion: str, sample: dict, compute_stats, max_lines: int = 40) -> None:
    stats = compute_stats(completion, sample)

    def _clip(text):
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text
        return "\n".join(lines[:max_lines] + [f"... (+{len(lines) - max_lines} lines)"])

    print("=" * 78)
    print(sample.get("prompt", ""))
    print("-" * 78)
    print("PREDICTION")
    print(_clip(completion.strip()))
    if "optimal_response" in sample:
        print("-" * 78)
        print("REFERENCE")
        print(_clip(sample["optimal_response"]))
    print("-" * 78)
    for label, value in stats.items():
        print(f"{label:<18}: {value}")
    print("=" * 78)


def save_report(report: dict, directory) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in report["method"].lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    path = directory / f"report_{slug}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved report to {path}")
    return path


def load_report(path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
