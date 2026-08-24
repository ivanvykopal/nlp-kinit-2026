"""Training-curve and metric-by-group plots, generic over any aggregate() shape."""
from __future__ import annotations

from pathlib import Path


def history_from_log(log_history: list[dict], keys=None) -> dict:
    """Turn a `Trainer.state.log_history`-shaped list into `{"steps", "series"}`.

    Not every key is logged at every step (e.g. `eval_loss` only at eval
    steps), so each series is padded with `None` for the steps it's missing.
    """
    steps = []
    series = {}
    for record in log_history:
        step = record.get("step")
        if step is None:
            continue
        wanted = keys if keys is not None else [k for k in record if k not in {"step", "epoch"}]
        values = {k: record.get(k) for k in wanted}
        if all(v is None for v in values.values()):
            continue
        steps.append(step)
        for key in set(series) | set(values):
            series.setdefault(key, [None] * (len(steps) - 1))
            series[key].append(values.get(key))
    return {"steps": steps, "series": series}


def plot_history(history: dict, keys: list[str], out_path, title="", ylabel="") -> None:
    import matplotlib.pyplot as plt

    plotted = False
    plt.figure(figsize=(7, 4.5))
    for key in keys:
        values = history["series"].get(key)
        if values is None:
            continue
        points = [(s, v) for s, v in zip(history["steps"], values) if v is not None]
        if not points:
            continue
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3,
                  label=key)
        plotted = True
    if not plotted:
        plt.close()
        print(f"No data for {list(keys)}; skipping {out_path}")
        return

    # In notebooks, `plt.show()` renders the figure inline below the cell before
    # we close it; under the non-interactive Agg backend it is a no-op, so the
    # saved file is unaffected when these run as a plain script.
    plt.xlabel("Training step")
    plt.ylabel(ylabel or "value")
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


def plot_metric_by_group(reports: list[dict], split: str, metric_key: str, out_path,
                          title="") -> None:
    import matplotlib.pyplot as plt

    plotted = False
    plt.figure(figsize=(7, 4.5))
    for report in reports:
        by_group = report["splits"].get(split, {}).get("by_group", {})
        if not by_group:
            continue
        items = sorted(by_group.items(), key=lambda kv: str(kv[0]))
        xs = [str(g) for g, _ in items]
        plt.plot(xs, [m.get(metric_key) for _, m in items], marker="o",
                  label=report["method"])
        plotted = True
    if not plotted:
        plt.close()
        print(f"No data for split {split!r}; skipping {out_path}")
        return
    plt.xlabel("Group")
    plt.ylabel(metric_key)
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
