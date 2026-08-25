"""End-to-end parity between openpi's 4.53 transformers patch and the 4.57 port.

The 4.53 fork shipped its patch by copying files over the installed
transformers package, so every import saw the patched classes. This package is
imported instead, which is why the wiring has to be checked rather than assumed:
anything still resolving through transformers.models.* now gets stock code.

Run twice, from two environments:

  # reference: transformers 4.53.2, original patch copied over transformers/
  git show 3fc057c:src/openpi/models_pytorch/gemma_pytorch.py > orig_gemma_pytorch.py
  for f in gemma/configuration_gemma gemma/modeling_gemma \
           paligemma/modeling_paligemma siglip/modeling_siglip; do
      git show 3fc057c:src/openpi/models_pytorch/transformers_replace/models/$f.py \
        > $SITE_PACKAGES/transformers/models/$f.py
  done
  python check_parity.py ref

  # test: current transformers, the vendored port
  python check_parity.py test

Both build PaliGemmaWithExpertModel with use_adarms=[False, True] (Pi0.5) in
bfloat16 and run every path the patch touches: the SigLIP bf16 cast and the
removed sqrt(hidden_size) image rescale, the prefix cache build, the repeated
denoise reads of that cache, and the interleaved training path. ref saves
weights, inputs and outputs; test reloads them and compares.

Three denoise steps, not one: the first step reads the correct width whether or
not the implementation writes to the cache, so a single step cannot tell a
read-only implementation from a writing one.

This supersedes equiv.py, which compared GemmaModel alone on the no-cache path
in float32 and so missed a dropped element of the patch entirely.
"""

import argparse, importlib.util, sys, types
from pathlib import Path

import torch

HERE = Path(__file__).parent
BLOB = HERE / "parity_ref.pt"
# openpi's interleaved path hardcodes 1 * 8 * head_dim, so num_heads must be 8
W, MLP, HEADS, HEAD_DIM, DEPTH, KV = 128, 256, 8, 16, 2, 2
BATCH, PREFIX, SUFFIX = 2, 6, 3


def cfg_ns(**kw):
    ns = types.SimpleNamespace(width=W, mlp_dim=MLP, num_heads=HEADS,
                               head_dim=HEAD_DIM, depth=DEPTH, num_kv_heads=KV)
    ns.__dict__.update(kw)
    return ns


VOCAB = 1024


def install_small_config_mapping(mod):
    """Keep the harness model small without touching gemma_pytorch's own code.

    gemma_pytorch hardcodes projection_dim=2048 and vocab_size=257152, which
    would force a 2048-wide text stack and a 526M-parameter embedding table.
    The config factory shrinks what it can; the constructor wrapper fixes the
    hardcoded fields in between gemma_pytorch setting them and the model being
    built. Swapping the config __class__ instead would break AutoModel, which
    resolves the vision tower by exact class.
    """
    from transformers.models.auto import CONFIG_MAPPING as REAL

    def small_paligemma():
        cfg = REAL["paligemma"]()
        v = cfg.vision_config
        v.hidden_size, v.num_hidden_layers, v.num_attention_heads = 64, 2, 4
        v.image_size, v.patch_size, v.num_channels = 32, 16, 3
        return cfg

    mod.CONFIG_MAPPING = {"paligemma": small_paligemma, "gemma": REAL["gemma"]}

    real_cls = mod.PaliGemmaForConditionalGeneration

    def build(config=None, **kw):
        config.vision_config.projection_dim = W    # must match the text width
        config.text_config.vocab_size = VOCAB
        config.image_token_index = VOCAB - 1
        return real_cls(config=config, **kw)

    mod.PaliGemmaForConditionalGeneration = build


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["ref", "test"])
ap.add_argument("--openpi-src", default=str(HERE.parents[2]))
args = ap.parse_args()

import transformers
torch.manual_seed(0)

if args.mode == "ref":
    mod = load_module(HERE / "orig_gemma_pytorch.py", "orig_gemma_pytorch")
else:
    sys.path.insert(0, args.openpi_src)
    import openpi.models_pytorch.gemma_pytorch as mod

install_small_config_mapping(mod)

model = mod.PaliGemmaWithExpertModel(
    vlm_config=cfg_ns(), action_expert_config=cfg_ns(),
    use_adarms=[False, True], precision="bfloat16",
).eval()

# the patch is only meaningful if AdaRMS actually built on the expert
expert_norm = model.gemma_expert.model.layers[0].input_layernorm
adarms_live = hasattr(expert_norm, "dense") and expert_norm.dense is not None
vlm_norm = model.paligemma.model.language_model.layers[0].input_layernorm
vlm_adarms = hasattr(vlm_norm, "dense") and vlm_norm.dense is not None

print(f"[{args.mode}] transformers {transformers.__version__}  torch {torch.__version__}")
print(f"[{args.mode}] expert AdaRMS live: {adarms_live}   vlm AdaRMS live: {vlm_adarms}")
assert adarms_live and not vlm_adarms, "AdaRMS wiring differs from Pi0.5"

if args.mode == "ref":
    torch.manual_seed(1)
    inputs = {
        "pixel": torch.randn(BATCH, 3, 32, 32),
        "prefix": torch.randn(BATCH, PREFIX, W),
        "suffix": torch.randn(BATCH, SUFFIX, W),
        "cond": torch.randn(BATCH, W),
    }
    torch.save({"state": model.state_dict(), "inputs": inputs}, BLOB)
else:
    blob = torch.load(BLOB, weights_only=False)
    missing, unexpected = model.load_state_dict(blob["state"], strict=False)
    assert not [k for k in missing if "embed_tokens" not in k], f"missing: {missing[:5]}"
    assert not unexpected, f"unexpected: {unexpected[:5]}"
    inputs = blob["inputs"]

bf16 = torch.bfloat16
pixel = inputs["pixel"].to(bf16)
prefix = inputs["prefix"].to(bf16)
suffix = inputs["suffix"].to(bf16)
# AdaRMS's dense sits inside input_layernorm, which stays float32 under
# to_bfloat16_for_selected_params, so the conditioning vector is float32.
cond = inputs["cond"].float()

results = {}
with torch.no_grad():
    # A. SigLIP + projector (the bf16 cast and the removed sqrt rescale)
    results["A_embed_image"] = model.embed_image(pixel)

    # B. prefix with cache (sample_actions)
    mask_p = torch.zeros(BATCH, 1, PREFIX, PREFIX, dtype=bf16)
    pos_p = torch.arange(PREFIX).unsqueeze(0).expand(BATCH, -1)
    (pre_out, _), cache = model.forward(
        attention_mask=mask_p, position_ids=pos_p, past_key_values=None,
        inputs_embeds=[prefix, None], use_cache=True, adarms_cond=[None, None])
    results["B_prefix"] = pre_out

    # C. denoise steps re-reading that cache (use_cache=False). sample_actions
    # runs 10 of these against one cache, so a single step is not enough: the
    # first step reads the right width either way, and only a later one sees a
    # cache that a writing implementation has grown.
    mask_s = torch.zeros(BATCH, 1, SUFFIX, PREFIX + SUFFIX, dtype=bf16)
    pos_s = torch.arange(PREFIX, PREFIX + SUFFIX).unsqueeze(0).expand(BATCH, -1)
    for step in range(3):
        (_, suf_out), _ = model.forward(
            attention_mask=mask_s, position_ids=pos_s, past_key_values=cache,
            inputs_embeds=[None, suffix], use_cache=False, adarms_cond=[None, cond])
        results[f"C_denoise{step}"] = suf_out

    # D. interleaved prefix+suffix (training / compute_loss)
    total = PREFIX + SUFFIX
    mask_d = torch.zeros(BATCH, 1, total, total, dtype=bf16)
    pos_d = torch.arange(total).unsqueeze(0).expand(BATCH, -1)
    (_, d_suf), _ = model.forward(
        attention_mask=mask_d, position_ids=pos_d, past_key_values=None,
        inputs_embeds=[prefix, suffix], use_cache=False, adarms_cond=[None, cond])
    results["D_interleaved"] = d_suf

results["C_cache_len"] = torch.tensor(float(cache.get_seq_length()))

if args.mode == "ref":
    blob = torch.load(BLOB, weights_only=False)
    blob["outputs"] = {k: v.float().clone() for k, v in results.items()}
    torch.save(blob, BLOB)
    for k, v in results.items():
        print(f"[ref] {k:16s} {tuple(v.shape)}")
    print("\nreference saved")
else:
    ref = torch.load(BLOB, weights_only=False)["outputs"]
    worst = 0.0
    for k in results:
        a, b = ref[k].float(), results[k].float()
        assert a.shape == b.shape, f"{k}: shape {a.shape} vs {b.shape}"
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print(f"[test] {k:16s} {tuple(b.shape)}  max|diff| = {d:.3e}")
    print(f"\nworst max|diff| across all paths = {worst:.3e}")
    print("PARITY OK" if worst == 0.0 else f"DIFFERS (worst {worst:.3e})")
