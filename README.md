# Fine-tuning LLMs (SFT, RLVR, LoRA) - Intro + Hands-On

## KInIT Natural Language Processing Summer School 2026

**When:** 2nd - 4th September 2026

**Where:** KInIT offices at The Spot, 6th floor, Sky Park Offices, Bottova 7939/2A, Bratislava

**Tutorial:** 3rd September 2026 (15:05 - 16:40)

**Link:** [https://kinit.sk/event/nlp-school/](https://kinit.sk/event/nlp-school/)

---

This repository contains the materials for the **Fine-tuning LLMs (SFT, RLVR, LoRA)** tutorial at the KInIT Natural Language Processing Summer School 2026. The session introduces how to adapt large language models to specific tasks and preferences, covering three complementary approaches:

- **Supervised Fine-Tuning (SFT)** — teaching the model to follow instructions by training on labelled input/output pairs.
- **Reinforcement Learning with Verifiable Rewards (RLVR)** — aligning the model to objective, checkable outcomes rather than human-annotated preferences.
- **Parameter-Efficient Fine-Tuning (PEFT) / LoRA** — adapting a model by training only a small set of injected parameters, making fine-tuning tractable on consumer hardware.

Each topic is accompanied by hands-on examples that can be run directly in Google Colab.

## Examples overview

The examples are grouped by topic, mirroring the `examples/` directory structure.
Each notebook opens with a short theory section, so a reader who missed the talk
can still follow it.

### Supervised Fine-Tuning (SFT)

| Topic | Google Colab | Jupyter Notebook |
| --- | --- | --- |
| Full fine-tuning (FFT) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/nlp-kinit-2026/blob/main/examples/sft/fft.ipynb) | [fft.ipynb](examples/sft/fft.ipynb) |

### Reinforcement Learning with Verifiable Rewards (RLVR)

We use **GSPO** (Group Sequence Policy Optimization), which keeps GRPO's
group-relative advantage but computes one length-normalised importance
ratio per sequence instead of one per token — a better match for our
whole-trajectory reward.

| Topic | Google Colab | Jupyter Notebook |
| --- | --- | --- |
| GSPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/nlp-kinit-2026/blob/main/examples/rl/gspo.ipynb) | [gspo.ipynb](examples/rl/gspo.ipynb) |

### Parameter-Efficient Fine-Tuning (PEFT)

| Topic | Google Colab | Jupyter Notebook |
| --- | --- | --- |
| LoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/nlp-kinit-2026/blob/main/examples/peft/lora.ipynb) | [lora.ipynb](examples/peft/lora.ipynb) |

## Further resources

For a deeper, hands-on dive into parameter-efficient fine-tuning methods specifically — covering sequential adapters, cross-lingual transfer, QLoRA, adapter fusion, and prompt tuning — see the materials from last year's edition of the summer school:

👉 [**PEFT - Intro + Hands-On (KInIT NLP Summer School 2025)**](https://github.com/ivanvykopal/peft-kinit-2025)
