"""Stats aggregation and model/split evaluation, generic over any task's compute_stats."""
from __future__ import annotations

import dataclasses
from typing import Callable

from lab.generation import generate_completions


@dataclasses.dataclass(frozen=True)
class FlatTask:
    """The two task-specific functions evaluation needs: how to turn a sample into a
    prompt, and how to score what came back."""
    compute_stats: Callable   # (completion_text, sample) -> dict
    to_chat_prompt: Callable  # (sample) -> list[dict]


def aggregate(stats: list[dict]) -> dict:
    """Average every numeric/bool field found across `stats`.

    A field missing from some entries, or `None` in some entries, is
    averaged only over the entries where it is present and not `None`.
    Always includes `"n"`, the number of entries aggregated.
    """
    n = len(stats)
    if n == 0:
        raise ValueError("Cannot aggregate an empty list of stats.")

    keys = set()
    for s in stats:
        keys.update(s.keys())

    result = {"n": n}
    for key in keys:
        if key == "n":  # Skip the "n" field from stats; "n" in result is always the count
            continue
        values = [s[key] for s in stats if s.get(key) is not None]
        if not values:
            result[key] = None
            continue
        if all(isinstance(v, bool) for v in values):
            result[key] = sum(values) / len(values)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            result[key] = sum(values) / len(values)
        # non-numeric fields (e.g. a string) are left out of the aggregate on purpose
    return result


def evaluate_split(model, tokenizer, samples, name, task,
                    group_key=None, max_new_tokens=512, batch_size=16, keep_completions=True):
    print(f"Evaluating '{name}' ({len(samples)} instances)...")
    completions = generate_completions(model, tokenizer, samples, task.to_chat_prompt,
                                        max_new_tokens=max_new_tokens, batch_size=batch_size)
    stats = [task.compute_stats(c, s) for c, s in zip(completions, samples)]

    by_group = {}
    if group_key is not None:
        groups = sorted({group_key(s) for s in samples}, key=str)
        for group in groups:
            subset = [st for st, s in zip(stats, samples) if group_key(s) == group]
            by_group[group] = aggregate(subset)

    return {
        "name": name,
        "overall": aggregate(stats),
        "by_group": by_group,
        "completions": list(completions) if keep_completions else [],
    }


def evaluate_model(model, tokenizer, splits: dict, method, task,
                    group_key=None, max_new_tokens=512, batch_size=16):
    report = {"method": method, "splits": {}}
    for name, samples in splits.items():
        report["splits"][name] = evaluate_split(
            model, tokenizer, samples, name, task,
            group_key=group_key, max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
    return report
