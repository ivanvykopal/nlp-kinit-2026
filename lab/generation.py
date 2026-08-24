"""Batched chat generation shared by every notebook, independent of any task."""
from __future__ import annotations

import torch


@torch.no_grad()
def generate_from_chats(model, tokenizer, chats, max_new_tokens=512, batch_size=16,
                         show_progress=True, do_sample=False, temperature=1.0):
    """Decode one completion per chat, in left-padded batches. Greedy by default."""
    was_training = model.training
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    completions = []
    try:
        for start in range(0, len(chats), batch_size):
            batch = chats[start: start + batch_size]
            texts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in batch
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                                add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = outputs[:, inputs["input_ids"].shape[1]:]
            completions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            if show_progress:
                print(f"  generated {min(start + batch_size, len(chats))}/{len(chats)}", end="\r")
    finally:
        tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()
    if show_progress:
        print()
    return completions


def generate_completions(model, tokenizer, samples, to_chat_prompt, max_new_tokens=512,
                          batch_size=16, show_progress=True):
    """Generate one completion per dataset sample, using the task's own `to_chat_prompt`."""
    return generate_from_chats(
        model, tokenizer, [to_chat_prompt(s) for s in samples],
        max_new_tokens=max_new_tokens, batch_size=batch_size, show_progress=show_progress,
    )
