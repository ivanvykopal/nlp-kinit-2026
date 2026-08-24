"""Mid-training metric probe: periodically measure a small fixed subset."""
from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback

from lab.evaluation import aggregate
from lab.generation import generate_completions


class MetricProbe(TrainerCallback):
    """Periodically evaluate a fixed subset and record whichever metrics are asked for."""

    def __init__(self, samples, tokenizer, task,
                 metrics=("solved", "optimal"), every=20, max_new_tokens=512, batch_size=16):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.task = task
        self.metrics = tuple(metrics)
        self.every = every
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.steps: list[int] = []
        self.history: dict[str, list[float]] = {m: [] for m in self.metrics}

    def _probe(self, model, step):
        completions = generate_completions(model, self.tokenizer, self.samples,
                                            self.task.to_chat_prompt, max_new_tokens=self.max_new_tokens,
                                            batch_size=self.batch_size, show_progress=False)
        stats = [self.task.compute_stats(c, s) for c, s in zip(completions, self.samples)]
        metrics = aggregate(stats)
        self.steps.append(step)
        summary = []
        for name in self.metrics:
            value = metrics.get(name)
            self.history[name].append(value)
            summary.append(f"{name} {value:.2f}" if value is not None else f"{name} n/a")
        print(f"  [probe @ step {step}] " + "  ".join(summary) + f"  (n={metrics['n']})")
        return metrics

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
        for name, values in self.history.items():
            plt.plot(self.steps, values, marker="o", label=name)
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
