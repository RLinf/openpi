"""Assert the action expert reads the prefix cache without mutating it.

pi0_pytorch.sample_actions runs the prefix once with use_cache=True to build
a DynamicCache, then denoise_step re-reads that same cache with
use_cache=False on each of its steps. Upstream 4.57 GemmaAttention.forward
does not declare use_cache, so an unpatched signature swallows the flag into
**kwargs and every step appends the suffix K/V to the prefix cache instead of
reading past it. The cache then grows step over step, and the mismatched
states surface as "expected scalar type BFloat16 but found Float" in the
attention matmul rather than anywhere near the cause.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from transformers.cache_utils import DynamicCache  # noqa: E402

from openpi.models_pytorch.transformers_replace.models.gemma import (  # noqa: E402
    configuration_gemma as C,
    modeling_gemma as G,
)

BATCH, PREFIX, SUFFIX, STEPS = 2, 6, 3, 3

cfg = C.GemmaConfig(
    vocab_size=128, hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, max_position_embeddings=64,
)
cfg._attn_implementation = "eager"
cfg.use_adarms = True
cfg.adarms_cond_dim = 32

torch.manual_seed(0)
model = G.GemmaModel(cfg).to(torch.bfloat16).eval()

prefix = torch.randn(BATCH, PREFIX, cfg.hidden_size, dtype=torch.bfloat16)
suffix = torch.randn(BATCH, SUFFIX, cfg.hidden_size, dtype=torch.bfloat16)
cond = torch.randn(BATCH, cfg.adarms_cond_dim, dtype=torch.bfloat16)

with torch.no_grad():
    cache = model(
        inputs_embeds=prefix,
        attention_mask=torch.ones(BATCH, PREFIX, dtype=torch.long),
        past_key_values=DynamicCache(),
        use_cache=True,
        adarms_cond=cond,
    ).past_key_values

assert cache.get_seq_length() == PREFIX, cache.get_seq_length()

position_ids = torch.arange(PREFIX, PREFIX + SUFFIX).unsqueeze(0).expand(BATCH, -1)
mask = torch.ones(BATCH, 1, SUFFIX, PREFIX + SUFFIX, dtype=torch.bfloat16)

outputs = []
for step in range(STEPS):
    with torch.no_grad():
        outputs.append(
            model(
                inputs_embeds=suffix,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=False,
                adarms_cond=cond,
            ).last_hidden_state
        )
    assert cache.get_seq_length() == PREFIX, (
        f"step {step} grew the prefix cache to {cache.get_seq_length()}; "
        "use_cache=False was ignored"
    )

assert outputs[0].shape == (BATCH, SUFFIX, cfg.hidden_size), outputs[0].shape
assert outputs[0].dtype == torch.bfloat16, outputs[0].dtype
assert torch.isfinite(outputs[0]).all()
for step, out in enumerate(outputs[1:], start=1):
    assert torch.equal(outputs[0], out), f"step {step} diverged from step 0"

print(
    f"OK: {STEPS} denoise steps read {PREFIX} cached prefix tokens, "
    "cache unchanged, outputs identical, bfloat16 preserved"
)
