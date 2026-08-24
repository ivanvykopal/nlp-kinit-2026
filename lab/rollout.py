"""Closed-loop (multi-step) rollout: an environment loop over a per-step policy.

Generic over any task that can supply: an initial state per instance, a way
to render a state as a chat prompt, a move parser, a legality-checked step
function, a solved check, and a distance-to-goal function.
`lab/evaluation.py` covers one-shot generation; this covers a policy making
a sequence of decisions against an environment that talks back. Never
imports a task module by name.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

from transformers import TrainerCallback

from lab.generation import generate_from_chats


@dataclasses.dataclass(frozen=True)
class StepwiseTask:
    """Everything `rollout_batch` needs to step one task's episodes.

    Gathering these eight functions into one object is what lets a notebook
    call `rollout_report(splits, move_fn, TASK)` on a single line instead of
    threading nine positional callables through every call site.

    Every function takes `(state, instance)`, except `apply_move(state, move)
    -> (state, legal)`, `parse_move(text) -> move | None`, `initial_state
    (instance) -> state`, and `optimal_moves(instance) -> int`.
    """
    initial_state: Callable
    state_prompt: Callable
    is_solved: Callable
    distance: Callable
    optimal_move: Callable
    optimal_moves: Callable
    parse_move: Callable
    apply_move: Callable


def rollout_batch(instances, move_fn, task, budget_factor=2.0, verbose=True) -> list:
    """Step every instance's episode in lockstep; one batched generation per tick.

    `move_fn(prompts: list[list[dict]]) -> list[str]` generates one completion
    per *still-active* episode. `task` is a `StepwiseTask` bundle -- see its
    docstring for the contract each of its eight functions must satisfy.

    Returns one dict per instance: `{"solved", "steps", "budget",
    "legal_moves", "illegal_moves", "unparsed", "improving_moves",
    "optimal_moves", "distance_at_end", "trace", "looped"}`.
    """
    n = len(instances)
    states = [task.initial_state(inst) for inst in instances]
    opt = [task.optimal_moves(inst) for inst in instances]
    budgets = [int(o * budget_factor) for o in opt]
    steps = [0] * n
    legal_moves = [0] * n
    illegal_moves = [0] * n
    unparsed = [0] * n
    improving_moves = [0] * n
    traces = [[] for _ in range(n)]
    visited = [{repr(states[i])} for i in range(n)]
    revisited = [False] * n

    max_budget = max(budgets) if budgets else 0
    for tick in range(max_budget):
        active = [i for i in range(n)
                  if not task.is_solved(states[i], instances[i]) and steps[i] < budgets[i]]
        if not active:
            break
        prompts = [task.state_prompt(states[i], instances[i]) for i in active]
        completions = move_fn(prompts)
        for i, completion in zip(active, completions):
            steps[i] += 1
            move = task.parse_move(completion)
            if move is None:
                unparsed[i] += 1
                illegal_moves[i] += 1
                shown = " ".join(completion.split())[:10]
                traces[i].append(f"?{shown}")
            else:
                before = task.distance(states[i], instances[i])
                new_state, legal = task.apply_move(states[i], move)
                traces[i].append(("" if legal else "x") + str(move))
                if legal:
                    legal_moves[i] += 1
                    states[i] = new_state
                    if task.distance(states[i], instances[i]) < before:
                        improving_moves[i] += 1
                else:
                    illegal_moves[i] += 1
            state_key = repr(states[i])
            if state_key in visited[i]:
                revisited[i] = True
            visited[i].add(state_key)
        if verbose:
            print(f"  step {tick + 1}/{max_budget}: {len(active)} episodes active", end="\r")
    if verbose:
        print(" " * 60, end="\r")

    results = []
    for i, inst in enumerate(instances):
        results.append({
            "solved": task.is_solved(states[i], inst),
            "steps": steps[i],
            "budget": budgets[i],
            "legal_moves": legal_moves[i],
            "illegal_moves": illegal_moves[i],
            "unparsed": unparsed[i],
            "improving_moves": improving_moves[i],
            "optimal_moves": opt[i],
            "distance_at_end": task.distance(states[i], inst),
            "trace": traces[i],
            "looped": revisited[i] and not task.is_solved(states[i], inst),
        })
    return results


def decision_accuracy(states, move_fn, task) -> dict:
    """Top-1/'improving'/illegal/unparsed rate of `move_fn` against `task.optimal_move`.

    `states` is a list of `(state, instance)` pairs. Always run this before
    `rollout_batch` -- it's one generation per state instead of a full
    episode per state, and it tells you whether a closed loop can possibly
    succeed: at per-decision "improving" accuracy `p`, distance-to-goal
    drifts by `(1 - 2p)` per step, so `p < 0.5` cannot solve at any budget.
    """
    if not states:
        raise ValueError("no decision states to score")
    prompts = [task.state_prompt(state, inst) for state, inst in states]
    completions = move_fn(prompts)

    top1 = illegal = unparsed = improving = 0
    for (state, inst), completion in zip(states, completions):
        want = task.optimal_move(state, inst)
        got = task.parse_move(completion)
        if got is None:
            unparsed += 1
            continue
        if got == want:
            top1 += 1
        new_state, legal = task.apply_move(state, got)
        if not legal:
            illegal += 1
            continue
        if task.distance(new_state, inst) < task.distance(state, inst):
            improving += 1

    total = len(states)
    improving_rate = improving / total
    return {
        "n": total,
        "top1": top1 / total,
        "improving": improving_rate,
        "illegal": illegal / total,
        "unparsed": unparsed / total,
        "drift": 1.0 - 2.0 * improving_rate,
    }


def summarize_episodes(results: list) -> dict:
    """Aggregate a list of `rollout_batch`-shaped result dicts."""
    n = len(results)
    if n == 0:
        raise ValueError("no episodes to summarise")
    solved = [r for r in results if r["solved"]]
    unsolved = [r for r in results if not r["solved"]]
    total_steps = sum(r["steps"] for r in results) or 1
    return {
        "n": n,
        "solved_rate": len(solved) / n,
        "optimal_rate": sum(1 for r in results if r["solved"] and r["steps"] == r["optimal_moves"]) / n,
        "avg_illegal_moves": sum(r["illegal_moves"] for r in results) / n,
        "avg_unparsed": sum(r["unparsed"] for r in results) / n,
        "improving_move_rate": sum(r["improving_moves"] for r in results) / total_steps,
        "avg_distance_closed": sum(
            1.0 - r["distance_at_end"] / max(r["optimal_moves"], 1) for r in results
        ) / n,
        "looped_rate": (sum(r["looped"] for r in unsolved) / len(unsolved)) if unsolved else 0.0,
    }


def rollout_report(eval_splits: dict, move_fn, task, group_key=None,
                    budget_factor=2.0, verbose=True) -> dict:
    """Run `rollout_batch` over every split and aggregate overall + by-group."""
    report = {}
    for name, instances in eval_splits.items():
        if verbose:
            print(f"Rolling out '{name}' ({len(instances)} episodes)...")
        results = rollout_batch(instances, move_fn, task,
                                 budget_factor=budget_factor, verbose=verbose)
        by_group = {}
        if group_key is not None:
            groups = sorted({group_key(inst) for inst in instances}, key=str)
            for group in groups:
                subset = [r for r, inst in zip(results, instances) if group_key(inst) == group]
                by_group[group] = summarize_episodes(subset)
        report[name] = {
            "overall": summarize_episodes(results), "by_group": by_group, "episodes": results,
        }
    return report


def print_rollout_report(report: dict, label: str) -> None:
    print()
    print("=" * 88)
    print(f"  {label}  (closed-loop rollout)")
    print("=" * 88)
    print(f"{'split':<16}{'group':>9}{'n':>5}{'solved':>9}{'good moves':>12}"
          f"{'dist closed':>13}{'looped':>9}")
    print("-" * 88)
    for name, data in report.items():
        for group, s in data["by_group"].items():
            print(f"{name:<16}{str(group):>9}{s['n']:>5}{s['solved_rate']:>9.3f}"
                  f"{s['improving_move_rate']:>12.3f}{s['avg_distance_closed']:>13.3f}"
                  f"{s['looped_rate']:>9.3f}")
        o = data["overall"]
        print(f"{'':<16}{'ALL':>9}{o['n']:>5}{o['solved_rate']:>9.3f}"
              f"{o['improving_move_rate']:>12.3f}{o['avg_distance_closed']:>13.3f}"
              f"{o['looped_rate']:>9.3f}")
        print("-" * 88)
    print("'good moves' = share of decisions that reduced the exact distance to the goal.")
    print("'dist closed' = 1.0 means finished; 0.0 means no closer than the start.")
    print("'looped' = share of UNSOLVED episodes stuck repeating one action after "
          "revisiting a state. Should be near zero with sampled decoding.")
    print("=" * 88)


class StepwiseMetricProbe(TrainerCallback):
    """Periodically run closed-loop episodes during training and record solved/looped rate.

    Sibling to `lab.probe.MetricProbe`, but for a per-step policy: a single
    generation cannot tell you whether the policy actually finishes puzzles,
    only `rollout_batch` can. Always samples (never greedy) -- a per-step
    prompt carries no history, so greedy decoding deterministically freezes
    the instant an episode revisits a state.
    """

    def __init__(self, instances, tokenizer, task, every=20, budget_factor=2.0,
                 temperature=0.7, max_new_tokens=8, batch_size=32):
        self.instances = list(instances)
        self.tokenizer = tokenizer
        self.task = task
        self.every = every
        self.budget_factor = budget_factor
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.steps: list = []
        self.solved_rate: list = []
        self.looped_rate: list = []

    def _probe(self, model, step):
        def move_fn(prompts):
            return generate_from_chats(model, self.tokenizer, prompts,
                                        max_new_tokens=self.max_new_tokens,
                                        batch_size=self.batch_size, show_progress=False,
                                        do_sample=True, temperature=self.temperature)

        results = rollout_batch(self.instances, move_fn, self.task,
                                 budget_factor=self.budget_factor, verbose=False)
        summary = summarize_episodes(results)
        self.steps.append(step)
        self.solved_rate.append(summary["solved_rate"])
        self.looped_rate.append(summary["looped_rate"])
        print(f"  [probe @ step {step}] solved {summary['solved_rate']:.2f}  "
              f"looped {summary['looped_rate']:.2f}  ({len(self.instances)} instances)")
        return summary

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._probe(model, int(state.global_step))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step % self.every == 0:
            self._probe(model, step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if not self.steps or self.steps[-1] != step:
            self._probe(model, step)

    def plot(self, out_path, title=""):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4.5))
        plt.plot(self.steps, self.solved_rate, marker="o", label="solved")
        plt.plot(self.steps, self.looped_rate, marker="s", label="looped")
        plt.xlabel("Training step")
        plt.ylabel("Rate on probe subset")
        plt.ylim(-0.05, 1.05)
        if title:
            plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved figure to {out_path}")
