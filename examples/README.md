# Examples

Six self-contained Colab notebooks for the Tower of Hanoi fine-tuning
tutorial. Each clones this repo at runtime to import `lab/` (generic
training/eval infra) and `tasks/hanoi.py` (the Hanoi verifier, dataset
formatting, and reward) — open a notebook's Colab badge to run it; no
local setup needed.

### Flat formulation — the whole solution in one generation

- `sft/fft.ipynb` — full fine-tuning
- `peft/lora.ipynb` — LoRA
- `rl/gspo.ipynb` — GSPO (run `sft/fft.ipynb` first; GSPO initializes from its checkpoint)

### Reference

- `RL_seminar_20260819.ipynb` — the earlier seminar notebook this tutorial was modelled on
