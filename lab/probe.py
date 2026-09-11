"""Mid-training metric probe: periodically measure a small fixed subset."""
from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback

from lab.evaluation import aggregate
from lab.generation import generate_completions


class MetricProbe(TrainerCallback):
    """Periodically evaluate a fixed subset and record whichever metrics are asked for."""

    def __init__(self, samples, tokenizer, task,
                 metrics=("solved", "optimal"), every=20, max_new_tokens=512, batch_size=16,
                 best_metric=None, best_dir=None):
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.task = task
        self.metrics = tuple(metrics)
        self.every = every
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.steps: list[int] = []
        self.history: dict[str, list[float]] = {m: [] for m in self.metrics}
        # Best-checkpoint selection: when `best_dir` is set, snapshot the model
        # every time `best_metric` reaches a new high on the probe subset, so a
        # run that peaks then drifts can be rolled back to its best step. With
        # no reference model to anchor the policy, this is the guardrail against
        # over-optimisation.
        self.best_metric = best_metric
        self.best_dir = Path(best_dir) if best_dir else None
        self.best_value: float = float("-inf")
        self.best_step: int | None = None

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
        self._maybe_save_best(model, metrics, step)
        return metrics

    def _maybe_save_best(self, model, metrics, step):
        if self.best_dir is None or self.best_metric is None:
            return
        value = metrics.get(self.best_metric)
        if value is None or value <= self.best_value:
            return
        self.best_value, self.best_step = value, step
        self.best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self.best_dir))
        self.tokenizer.save_pretrained(str(self.best_dir))
        print(f"    new best {self.best_metric}={value:.3f} @ step {step} -> saved to {self.best_dir}")

    def best_summary(self):
        """Where the peak-probe checkpoint was saved, or dir=None if none was."""
        recorded = self.best_dir is not None and self.best_step is not None
        return {
            "metric": self.best_metric,
            "value": self.best_value if recorded else None,
            "step": self.best_step,
            "dir": str(self.best_dir) if recorded else None,
        }

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
