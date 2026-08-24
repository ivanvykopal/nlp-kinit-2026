# %% [markdown]
# ## Task: Tower of Hanoi
#
# This module is the one place that knows what "Tower of Hanoi" means:
# how to parse a model's answer, how to check whether a sequence of moves
# actually solves the puzzle, how to turn a dataset row into a training
# example, and how to reward a rollout. Everything in `lab/` is generic —
# it never mentions a peg or a disk by name — and depends only on the
# functions below.
#
# Swapping this tutorial to a different verifiable task (Sudoku, a graph
# coloring problem, anything with a checker) means writing a new module
# shaped like this one; `lab/` would not change at all.
#
# A move is a plain `(source, target)` tuple of peg names — `("A", "C")`
# reads as "move the top disk of A onto C". No `Move` class: a tuple is
# already immutable, printable, and comparable, and one regex below is the
# entire move syntax.

# %%
"""Tower of Hanoi: parsing, solving, replaying, scoring, and reward."""
from __future__ import annotations

import random
import re
from typing import List

PEG_NAMES = ["A", "B", "C"]

# The whole move syntax: `A->C`, with optional spaces around the arrow.
MOVE_RE = re.compile(r"([{pegs}])\s*->\s*([{pegs}])".format(pegs="".join(PEG_NAMES)))

# %% [markdown]
# Two tiny helpers turn a move tuple into the text form the model reads and
# writes, and back — everything else in this module works with `(source,
# target)` tuples, so the text form only exists at the model boundary.

# %%
def move_to_text(move) -> str:
    return "{}->{}".format(*move)


def moves_to_text(moves) -> str:
    return "\n".join(move_to_text(m) for m in moves)


# %% [markdown]
# Turning model output back into moves needs two different levels of
# strictness. `parse_output` is used for scoring a whole flat solution: it
# walks every line and keeps a strict per-line count, so junk is counted
# rather than silently dropped. `first_move_in` is used by the per-step
# formulation, where the model is asked for exactly one move: it just finds
# the first well-formed move anywhere in the text and ignores the rest.

# %%
def parse_output(text: str):
    """Parse a whole completion into moves.

    Returns `(moves, n_lines, n_unparsed)`: every non-empty line is either a
    move or counted as unparseable — junk is counted, never silently
    dropped, so the caller can tell a clean list of moves from noisy output.
    """
    moves, n_lines, n_unparsed = [], 0, 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        n_lines += 1
        match = MOVE_RE.fullmatch(line)
        if match:
            moves.append(match.groups())
        else:
            n_unparsed += 1
    return moves, n_lines, n_unparsed


# %% [markdown]
# `first_move_in` is the lenient counterpart used by the per-step
# formulation. `completion_text` exists because a TRL completion arrives as
# either raw text or a list of chat turns, and every caller below wants
# plain text.

# %%
def first_move_in(text: str):
    """Lenient single-move parser: the first well-formed move anywhere in the text."""
    match = MOVE_RE.search(text)
    return match.groups() if match else None


def completion_text(completion) -> str:
    """A TRL completion is either raw text or a list of chat turns; flatten it."""
    if isinstance(completion, str):
        return completion
    return "".join(turn.get("content") or "" for turn in completion)


# %% [markdown]
# A reference solver, used two ways below: to build the optimal
# demonstrations the SFT notebooks train on, and to know the optimal move
# count for scoring — `2**n - 1`, the closed form for this recursion.

# %%
def solve_hanoi(n_disks, source, auxiliary, target) -> List[tuple]:
    """The unique optimal solution: move n-1 aside, move the bottom disk, move them back."""
    if n_disks == 0:
        return []
    return (
        solve_hanoi(n_disks - 1, source, target, auxiliary)
        + [(source, target)]
        + solve_hanoi(n_disks - 1, auxiliary, source, target)
    )


def optimal_move_count(n_disks: int) -> int:
    return 2 ** n_disks - 1


# %% [markdown]
# ### The verifier
#
# There is no environment *class* here, and no simulator either. A board is
# just a `dict` of `{peg: [disks]}`, top disk last in the list; `apply_move`
# is the single stepping primitive (used by the flat replay below *and* by
# the per-step formulation further down); and `replay` folds it over a list
# of moves. An illegal move ends the attempt — whatever follows it is not
# played, so it cannot count for anything.

# %%
def new_pegs(n_disks, source, auxiliary, target) -> dict:
    """All disks stacked on `source`; the other two pegs start empty."""
    return {source: list(range(n_disks, 0, -1)), auxiliary: [], target: []}


def is_legal(pegs: dict, move) -> bool:
    """Legal iff it moves the top disk of one peg onto a bigger (or empty) peg."""
    source, target = move
    if source == target or not pegs[source]:
        return False
    return not pegs[target] or pegs[source][-1] < pegs[target][-1]


# %% [markdown]
# `apply_move` is the single stepping primitive — used by the flat `replay`
# below and, further down, by the per-step formulation's own environment
# loop — so both training regimes agree on exactly what a legal move does.

# %%
def apply_move(pegs: dict, move) -> "tuple[dict, bool]":
    """Legality-checked step. Returns (new_pegs, legal); `pegs` is never mutated."""
    if not is_legal(pegs, move):
        return pegs, False
    source, target = move
    stepped = {peg: list(stack) for peg, stack in pegs.items()}
    stepped[target].append(stepped[source].pop())
    return stepped, True


def is_solved(pegs: dict, n_disks: int, target: str) -> bool:
    return len(pegs[target]) == n_disks


# %% [markdown]
# `replay` folds `apply_move` over a whole list of moves, for scoring a flat
# solution. An illegal move ends the attempt — whatever follows it is not
# played, so it cannot count for anything.

# %%
def replay(moves, n_disks, source, auxiliary, target) -> "tuple[dict, bool]":
    """Play `moves` from the start, stopping at the first illegal move.

    Returns the board it reached and whether an illegal move ended it.
    """
    pegs = new_pegs(n_disks, source, auxiliary, target)
    for move in moves:
        pegs, legal = apply_move(pegs, move)
        if not legal:
            return pegs, True
    return pegs, False


# %% [markdown]
# ### Loading the dataset
#
# The dataset lives on the HuggingFace Hub as five named splits — `train`,
# `grpo_train`, `heldout`, `train_instances`, `extrapolation`. One row looks
# like this:
#
# ```json
# {
#   "prompt": "...", "n_disks": 3, "source": "A", "auxiliary": "B", "target": "C",
#   "target_response": "...", "optimal_response": "...",
#   "corrupted": false, "corruption": null
# }
# ```

# %%
def load_split(name: str, repo_id: str):
    """Load one named split of the dataset from the HuggingFace Hub.

    Returns a `datasets.Dataset` — already iterable/indexable like a list of
    dicts (for `sum(r["corrupted"] for r in rows)`-style inspection) and
    already has `.map()` (for building the SFT/GSPO training format), so
    there's no separate "plain rows" vs. "Dataset" loading step.
    """
    from datasets import load_dataset

    return load_dataset(repo_id, split=name)


# %% [markdown]
# ### Grouping for the by-group breakdown and the mid-training probe
#
# Hanoi's natural grouping is disk count. A different task might group by
# difficulty, grid size, or not at all — `lab.evaluation`/`lab.probe` accept
# `group_key=None` and simply skip the breakdown when a task has nothing to
# group by.

# %%
def PROBE_GROUP_KEY(sample: dict) -> int:
    return sample["n_disks"]


PROBE_GROUP_VALUES = [4]


# %% [markdown]
# ### `compute_stats`: what a model's answer scores as
#
# `compute_stats` is what evaluation reports, and it deliberately reports
# only three things: **did it solve the puzzle**, **how many moves it wrote
# against the 2**n - 1 optimum**, and **was there anything in the output
# that was not a move**. Nothing else — every extra field is one more column
# to explain and one more thing that can quietly disagree with `solved`.
#
# `"solved"` (a bool) is the only field `lab/` requires; the rest is
# free-form, and `lab.evaluation.aggregate` averages whatever numeric/bool
# fields it finds without needing to know their names.

# %%
def compute_stats(prediction: str, sample: dict) -> dict:
    """Replay one completion and report what evaluation shows."""
    moves, n_lines, n_unparsed = parse_output(prediction)
    pegs, _illegal = replay(moves, sample["n_disks"], sample["source"],
                            sample["auxiliary"], sample["target"])
    return {
        "solved": is_solved(pegs, sample["n_disks"], sample["target"]),
        "total_moves": len(moves),
        "optimal_moves": optimal_move_count(sample["n_disks"]),
        "unparsed": n_unparsed > 0,
    }


# %% [markdown]
# ### From dataset row to training example
#
# Three small functions turn one dataset row into what a trainer expects: a
# prompt, and — for the two supervised methods — the answer to train on.

# %%
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert algorithmic problem solver. "
    "Solve the Tower of Hanoi puzzle optimally. "
    "Return ONLY one move per line in the format 'A->C'. "
    "Do not provide any explanation."
)

# %% [markdown]
# `to_chat_prompt` wraps `DEFAULT_SYSTEM_PROMPT` and the puzzle's own prompt
# text into the system+user half of a conversation. `to_sft_format` adds the
# assistant turn FFT/LoRA train on; `to_gspo_format` carries the puzzle's
# coordinates instead of any target, since GSPO never sees a correct answer —
# only enough information to check one.

# %%
def to_chat_prompt(sample: dict) -> List[dict]:
    """The prompt half of the conversation (system + user)."""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": sample["prompt"]},
    ]


def to_sft_format(sample: dict) -> dict:
    """Prompt/completion format for SFTTrainer (loss on the answer only)."""
    return {
        "prompt": to_chat_prompt(sample),
        "completion": [{"role": "assistant", "content": sample["target_response"]}],
    }


def to_gspo_format(sample: dict) -> dict:
    """GSPO conversational prompt plus the columns the reward function needs."""
    return {
        "prompt": to_chat_prompt(sample),
        "n_disks": sample["n_disks"],
        "source": sample["source"],
        "auxiliary": sample["auxiliary"],
        "target": sample["target"],
    }


# %% [markdown]
# ## Reward (GSPO only)
#
# Three components, in a total range of `[-0.7, +2.0]`, logged separately by
# TRL so a stalled run shows which one stopped improving. None of them grows
# with the number of moves produced — that is the reward-hacking guard from
# the theory section: a reward of the form `+1 per legal move` is maximized
# by shuffling disks legally forever and never solving anything, and that
# policy is perfectly stable, because TRL trains on truncated rollouts by
# default.
#
# Before trusting this reward, rank a handful of hand-written trajectories
# by hand — an empty answer should score worse than illegal moves, illegal
# moves worse than partial progress, and "solved but rambling" worse than
# "solved optimally." `tests/tasks/test_hanoi_reward.py` pins exactly this
# ordering.

# %%
FORMAT_WEIGHT = 0.2
EMPTY_PENALTY = 0.3
PROGRESS_WEIGHT = 0.5
ILLEGAL_PENALTY = 0.25  # below EMPTY_PENALTY: a wrong attempt beats refusing to answer
SOLVED_BONUS = 1.0
OPTIMAL_BONUS = 0.5
OVERSHOOT_WEIGHT = 0.15
# How many multiples of the optimal move count are still "wasteful" rather
# than "rambling".
OVERSHOOT_TOLERANCE = 3.0

# %% [markdown]
# Partial credit needs to know *how far* the model got, which is more than
# `compute_stats` reports, so the reward does its own replay here and keeps
# those extra numbers out of the evaluation table.

# %%
def reward_stats(prediction: str, sample: dict) -> dict:
    """Everything the reward needs: `compute_stats` plus how far the model got."""
    n_disks = sample["n_disks"]
    moves, n_lines, n_unparsed = parse_output(prediction)
    pegs, illegal = replay(moves, n_disks, sample["source"], sample["auxiliary"],
                           sample["target"])

    placed = 0
    for index, disk in enumerate(pegs[sample["target"]]):
        if disk != n_disks - index:  # correctly stacked, counted from the bottom
            break
        placed += 1

    return {
        "n_lines": n_lines,
        "n_unparsed": n_unparsed,
        "illegal": illegal,
        "progress_fraction": placed / n_disks,
        "solved": placed == n_disks,
        "total_moves": len(moves),
        "optimal_moves": optimal_move_count(n_disks),
    }


# %% [markdown]
# Two dense components, checked with the reward as a whole: `format_score`
# penalizes noise, `progress_score` gives partial credit for disks placed
# minus a penalty for an illegal move.

# %%
def format_score(stats: dict) -> float:
    """Penalize noise; saying nothing at all is the worst option."""
    if stats["n_lines"] == 0:
        return -EMPTY_PENALTY
    return -FORMAT_WEIGHT * stats["n_unparsed"] / stats["n_lines"]


def progress_score(stats: dict) -> float:
    """Dense partial credit for disks placed, minus a flat penalty for slipping."""
    score = PROGRESS_WEIGHT * stats["progress_fraction"]
    if stats["illegal"]:
        score -= ILLEGAL_PENALTY
    return score


# %% [markdown]
# `solved_score` is the sparse half of the reward: nothing until the puzzle
# is actually solved, then a bonus, plus a small penalty for rambling well
# past the optimal move count. `total_score` just adds the three components —
# TRL logs each one separately, so a run that stalls shows *which* component
# stopped improving.

# %%
def solved_score(stats: dict) -> float:
    """Sparse bonus for solving, plus a bounded penalty for rambling past the optimum."""
    extra_moves = max(0, stats["total_moves"] - stats["optimal_moves"])
    score = -OVERSHOOT_WEIGHT * min(1.0, extra_moves / (OVERSHOOT_TOLERANCE * stats["optimal_moves"]))
    if stats["solved"]:
        score += SOLVED_BONUS
        if stats["total_moves"] == stats["optimal_moves"]:
            score += OPTIMAL_BONUS
    return score


def total_score(stats: dict) -> float:
    return format_score(stats) + progress_score(stats) + solved_score(stats)


# %% [markdown]
# TRL forwards every extra dataset column to the reward function as a
# kwarg, so `n_disks`/`source`/`auxiliary`/`target` arrive as lists aligned
# with `completions`, and it names each logged reward column after the
# function it came from — hence three thin callbacks around `batch_stats`
# rather than one reward function.

# %%
def batch_stats(completions, n_disks, source, auxiliary, target) -> List[dict]:
    return [
        reward_stats(completion_text(c),
                     {"n_disks": n, "source": src, "auxiliary": aux, "target": dst})
        for c, n, src, aux, dst in zip(completions, n_disks, source, auxiliary, target)
    ]


def format_reward(completions, n_disks, source, auxiliary, target, **kwargs):
    return [format_score(s) for s in batch_stats(completions, n_disks, source, auxiliary, target)]


def progress_reward(completions, n_disks, source, auxiliary, target, **kwargs):
    return [progress_score(s) for s in batch_stats(completions, n_disks, source, auxiliary, target)]


def solved_reward(completions, n_disks, source, auxiliary, target, **kwargs):
    return [solved_score(s) for s in batch_stats(completions, n_disks, source, auxiliary, target)]


REWARD_FUNCS = [format_reward, progress_reward, solved_reward]

# %% [markdown]
# ## Per-step formulation
#
# Everything above asks for a whole solution in one generation (the "flat"
# formulation). A prior investigation found that this formulation fails to
# extrapolate to more disks not because errors accumulate, but because the
# model learns a **length prior**: trained on solutions of at most 31 moves
# (5 disks), it emits exactly 31 legal moves on a 63-move (6-disk) puzzle and
# then stops. Raising the token budget cannot fix this — it isn't truncated,
# it chooses to stop.
#
# The fix: ask for **one move at a time**, conditioned on the current board.
# An environment loop applies the move and asks again. Output length becomes
# ~3 tokens regardless of disk count, so there is no length to memorize.

# %%
STEPWISE_SYSTEM_PROMPT = (
    "You are solving the Tower of Hanoi. Peg contents are listed bottom to "
    "top, so the last number on a peg is the disk you may move. Larger "
    "disks may never sit on smaller ones. Reply with exactly one move in "
    "the format 'A->C' and nothing else."
)

EMPTY_PEG = "-"


def render_pegs(pegs: dict) -> str:
    """`A:6,4 | B:5,3,2 | C:1` — pegs always listed A, B, C, regardless of dict order."""
    parts = []
    for peg in PEG_NAMES:
        stack = pegs.get(peg, [])
        parts.append(f"{peg}:" + (",".join(str(d) for d in stack) if stack else EMPTY_PEG))
    return " | ".join(parts)


def state_prompt(pegs: dict, n_disks: int, target: str) -> str:
    return f"Disks: {n_disks}\n{render_pegs(pegs)}\nTarget peg: {target}\nNext move:"


def to_stepwise_chat_prompt(pegs: dict, n_disks: int, target: str) -> List[dict]:
    return [
        {"role": "system", "content": STEPWISE_SYSTEM_PROMPT},
        {"role": "user", "content": state_prompt(pegs, n_disks, target)},
    ]


# %% [markdown]
# ### Per-step formulation: the exact policy and potential function
#
# The next two functions answer the two questions the per-step formulation
# needs: what move should the policy make from here, and how close is this
# state to solved? The second is a **potential function** — a number that
# measures distance to the goal — which the reward section further below
# turns into per-move feedback: reward a move by how much the potential
# dropped. `_location` and `_third` below are small shared helpers.

# %%
def _location(pegs: dict, disk: int) -> str:
    for peg, stack in pegs.items():
        if disk in stack:
            return peg
    raise ValueError(f"disk {disk} is not on any peg: {pegs}")


def _third(a: str, b: str) -> str:
    return next(p for p in PEG_NAMES if p != a and p != b)


# %% [markdown]
# `optimal_next_move` and `remaining_moves` are the same recursion twice:
# "get disks 1..k onto the target peg", stated once as the first move to
# make and once as how many moves it takes. Both work from *any* legal
# configuration, not just the canonical start.

# %%
def optimal_next_move(pegs: dict, n_disks: int, target: str):
    """First move of an optimal completion from this configuration; `None` if solved."""
    def first_move(k, dst):
        if k == 0:
            return None
        location = _location(pegs, k)
        if location == dst:
            return first_move(k - 1, dst)
        spare = _third(location, dst)
        sub = first_move(k - 1, spare)
        return (location, dst) if sub is None else sub
    return first_move(n_disks, target)


def remaining_moves(pegs: dict, n_disks: int, target: str) -> int:
    """Exact optimal move count from this configuration — an admissible potential function."""
    def count(k, dst):
        if k == 0:
            return 0
        location = _location(pegs, k)
        if location == dst:
            return count(k - 1, dst)
        spare = _third(location, dst)
        return count(k - 1, spare) + 1 + (2 ** (k - 1) - 1)
    return count(n_disks, target)


# %% [markdown]
# ### Per-step formulation: building the dataset from the same data
#
# No new download, no new HF Hub split: `build_stepwise_rows` takes the flat
# `train`/`train_instances` rows already loaded elsewhere in this notebook —
# only their `n_disks`/`source`/`auxiliary`/`target` fields matter — and
# samples decision points from them: half from the optimal path, half from
# uniformly-random legal configurations, so the policy also learns to act
# after a mistake, not only along the golden trajectory.
# `corruption_probability` matches the flat dataset's 20% corrupted-label
# rate, so the supervised-noise story is unchanged.

# %%
ALL_MOVES = [(a, b) for a in PEG_NAMES for b in PEG_NAMES if a != b]


def _random_pegs(n_disks: int, rng: "random.Random") -> dict:
    pegs = {peg: [] for peg in PEG_NAMES}
    for disk in range(n_disks, 0, -1):
        pegs[rng.choice(PEG_NAMES)].append(disk)
    return pegs


# %% [markdown]
# `_optimal_path_states` gives `build_stepwise_rows` its "half from the
# optimal path" states: every board reached along `solve_hanoi`'s own
# solution, before the goal.

# %%
def _optimal_path_states(n_disks: int, source: str, auxiliary: str, target: str) -> List[dict]:
    pegs = new_pegs(n_disks, source, auxiliary, target)
    states = [pegs]
    for move in solve_hanoi(n_disks, source, auxiliary, target):
        pegs, _ = apply_move(pegs, move)
        states.append(pegs)
    return states


# %% [markdown]
# `build_stepwise_rows` samples decision points for one instance at a time,
# then turns each sampled state into a row. Two helpers name those two
# steps: `_sample_instance_states` does the half-on-path/half-random
# selection, `_stepwise_row` does the corruption draw that the entire
# supervised-noise story hinges on. Both take the same `random.Random`
# instance `build_stepwise_rows` created from `seed`, and call it in the
# same order the original single function did — the dataset is seeded, so
# reordering or adding an `rng` call would silently change every row after
# it.

# %%
def _sample_instance_states(n_disks: int, source: str, auxiliary: str, target: str,
                             states_per_instance: int, off_path_fraction: float,
                             rng: "random.Random") -> List[dict]:
    """States to build rows from: half sampled from the optimal path, half
    rejection-sampled from random legal configurations."""
    path = [s for s in _optimal_path_states(n_disks, source, auxiliary, target)
            if not is_solved(s, n_disks, target)]
    n_off = round(states_per_instance * off_path_fraction)
    n_on = states_per_instance - n_off
    sampled = [rng.choice(path) for _ in range(n_on)]
    attempts = 0
    while len(sampled) < states_per_instance and attempts < max(n_off, 1) * 20:
        attempts += 1
        candidate = _random_pegs(n_disks, rng)
        if not is_solved(candidate, n_disks, target):
            sampled.append(candidate)
    return sampled


# %% [markdown]
# `_stepwise_row` is where the 20% corrupted-label rate (matching the flat
# dataset's) actually happens: look up the optimal move, then with
# probability `corruption_probability` swap in a different, wrong one as
# the training label instead.

# %%
def _stepwise_row(state: dict, n_disks: int, source: str, auxiliary: str, target: str,
                   corruption_probability: float, rng: "random.Random") -> dict:
    """One training row: the optimal move at `state`, corrupted with the given probability."""
    best = optimal_next_move(state, n_disks, target)
    label, corrupted = best, False
    if corruption_probability > 0 and rng.random() < corruption_probability:
        label = rng.choice([m for m in ALL_MOVES if m != best])
        corrupted = True
    return {
        "n_disks": n_disks, "source": source, "auxiliary": auxiliary, "target": target,
        "pegs": state, "prompt": state_prompt(state, n_disks, target),
        "optimal_move": move_to_text(best), "target_move": move_to_text(label),
        "corrupted": corrupted,
    }


# %% [markdown]
# With both helpers in place, `build_stepwise_rows` itself is just a loop
# over instances and their sampled states.

# %%
def build_stepwise_rows(instances, states_per_instance: int, off_path_fraction: float,
                         corruption_probability: float, seed: int) -> List[dict]:
    """One row per (state, next-move) decision, derived from `instances` alone.

    `instances` is any iterable of dicts with `n_disks`/`source`/`auxiliary`/`target`
    — exactly the flat dataset's own row shape, so the flat `train`/`train_instances`
    splits can be passed here directly.
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    for inst in instances:
        n_disks, source, auxiliary, target = (
            inst["n_disks"], inst["source"], inst["auxiliary"], inst["target"]
        )
        sampled = _sample_instance_states(n_disks, source, auxiliary, target,
                                           states_per_instance, off_path_fraction, rng)
        for state in sampled:
            rows.append(_stepwise_row(state, n_disks, source, auxiliary, target,
                                       corruption_probability, rng))
    rng.shuffle(rows)
    return rows


# %% [markdown]
# `to_stepwise_sft_format` and `to_stepwise_gspo_format` mirror the flat
# dataset's `to_sft_format`/`to_gspo_format`, one decision point at a time:
# SFT gets the labeled move as its completion, GSPO gets only the board
# state and evaluates whatever it samples against `stepwise_reward`.

# %%
def to_stepwise_sft_format(row: dict) -> dict:
    """Prompt/completion format for SFTTrainer (loss on the answer only)."""
    return {
        "prompt": to_stepwise_chat_prompt(row["pegs"], row["n_disks"], row["target"]),
        "completion": [{"role": "assistant", "content": row["target_move"]}],
    }


def to_stepwise_gspo_format(row: dict) -> dict:
    """GSPO conversational prompt plus the columns `stepwise_reward` needs. No move label."""
    return {
        "prompt": to_stepwise_chat_prompt(row["pegs"], row["n_disks"], row["target"]),
        "pegs": row["pegs"], "n_disks": row["n_disks"], "target": row["target"],
    }


# %% [markdown]
# ### Per-step formulation: reward (GSPO only)
#
# `remaining_moves` is an exact distance-to-goal, so "did this move get
# closer" is a dense, perfectly verifiable signal. This turns RL here into a
# **contextual bandit** — one decision, one immediate reward, no credit
# assignment across an episode — a real simplification worth stating, not a
# free lunch.
#
# ```
#   +1.0  optimal move        -1.0  illegal move
#   -0.5  legal but wasteful  -1.5  not a move at all
# ```

# %%
def stepwise_move_reward(pegs: dict, n_disks: int, target: str, completion: str) -> float:
    move = first_move_in(completion)
    if move is None:
        return -1.5
    next_pegs, legal = apply_move(pegs, move)
    if not legal:
        return -1.0
    before = remaining_moves(pegs, n_disks, target)
    after = remaining_moves(next_pegs, n_disks, target)
    return 1.0 if after < before else -0.5


def stepwise_reward(completions, pegs, n_disks, target, **kwargs):
    """TRL reward callback: `pegs`/`n_disks`/`target` arrive as lists aligned with `completions`."""
    return [
        stepwise_move_reward(p, n, t, completion_text(c))
        for c, p, n, t in zip(completions, pegs, n_disks, target)
    ]


STEPWISE_REWARD_FUNCS = [stepwise_reward]
