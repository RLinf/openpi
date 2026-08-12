"""Cross-version equivalence check for the AdaRMS Gemma port.

Run twice:
  ref   — under transformers 4.53.2, loading openpi's forked modeling_gemma
  test  — under transformers 4.57.1, loading the rebased port

The reference run saves weights, inputs and outputs; the test run reloads the
same weights and inputs and compares. If the rebase altered Pi0.5's maths,
the max-abs difference is non-zero.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
BLOB = HERE / "equiv_ref.pt"

HIDDEN, COND, LAYERS, HEADS, KV, HEAD_DIM, VOCAB = 64, 32, 2, 4, 2, 16, 128


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(mod, cfg_cls):
    cfg = cfg_cls(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        head_dim=HEAD_DIM,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    cfg.use_adarms = True
    cfg.adarms_cond_dim = COND
    torch.manual_seed(0)
    model = mod.GemmaModel(cfg).to(torch.float32).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "test"])
    ap.add_argument("--module", required=True, help="modeling_gemma.py to load")
    ap.add_argument("--config", required=True, help="configuration_gemma.py to load")
    args = ap.parse_args()

    import transformers
    print(f"transformers {transformers.__version__}  torch {torch.__version__}")

    cfgmod = load_module(args.config, "eq_cfg")
    gemma = load_module(args.module, "eq_gemma")
    model = build(gemma, cfgmod.GemmaConfig)

    if args.mode == "ref":
        torch.manual_seed(1)
        ids = torch.randint(0, VOCAB, (2, 8))
        cond = torch.randn(2, COND)
        with torch.no_grad():
            out = model(input_ids=ids, adarms_cond=cond).last_hidden_state
        torch.save(
            {"state_dict": model.state_dict(), "ids": ids, "cond": cond, "out": out},
            BLOB,
        )
        print(f"saved reference: out{tuple(out.shape)} mean={out.mean():.6f} std={out.std():.6f}")
        print(f"  params: {sum(p.numel() for p in model.parameters())}")
        return 0

    blob = torch.load(BLOB, weights_only=False)
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print("  missing:", list(missing)[:6])
    if unexpected:
        print("  unexpected:", list(unexpected)[:6])

    with torch.no_grad():
        out = model(input_ids=blob["ids"], adarms_cond=blob["cond"]).last_hidden_state

    ref = blob["out"]
    if out.shape != ref.shape:
        print(f"SHAPE MISMATCH ref{tuple(ref.shape)} vs test{tuple(out.shape)}")
        return 1
    diff = (out - ref).abs()
    print(f"\nref  mean={ref.mean():.6f} std={ref.std():.6f}")
    print(f"test mean={out.mean():.6f} std={out.std():.6f}")
    print(f"max|diff| = {diff.max().item():.3e}")
    print(f"mean|diff| = {diff.mean().item():.3e}")
    ok = torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    print("\n" + ("NUMERICALLY EQUIVALENT" if ok else "DIVERGED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
