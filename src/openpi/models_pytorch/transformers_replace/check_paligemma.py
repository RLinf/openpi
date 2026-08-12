"""Assert PaliGemma builds the PATCHED Gemma/SigLIP, not the stock ones.

AutoModel.from_config resolves through transformers' global registry. Once
the Pi0.5 modules are imported instead of copied over transformers/, that
registry returns stock classes — the model still builds and runs, silently
without AdaRMS. This catches that.
"""
import sys, torch
sys.path.insert(0, "/tmp/claude-0/-mnt-public-hao-RLinf/4e06ea6b-9203-4ddb-a9c4-dbd2b8264362/scratchpad/openpi/src")
from transformers.models.auto import CONFIG_MAPPING
from openpi.models_pytorch.transformers_replace.models.paligemma import modeling_paligemma as P
from openpi.models_pytorch.transformers_replace.models.gemma import modeling_gemma as G
from openpi.models_pytorch.transformers_replace.models.siglip import modeling_siglip as S

cfg = CONFIG_MAPPING["paligemma"]()
cfg.text_config.hidden_size = 64
cfg.text_config.intermediate_size = 128
cfg.text_config.num_hidden_layers = 2
cfg.text_config.num_attention_heads = 4
cfg.text_config.num_key_value_heads = 2
cfg.text_config.head_dim = 16
cfg.text_config.use_adarms = True
cfg.text_config.adarms_cond_dim = 32
cfg.vision_config.hidden_size = 64
cfg.vision_config.intermediate_size = 128
cfg.vision_config.num_hidden_layers = 2
cfg.vision_config.num_attention_heads = 4

torch.manual_seed(0)
m = P.PaliGemmaForConditionalGeneration(cfg)
lm = m.model.language_model
vt = m.model.vision_tower
print(f"language_model : {type(lm).__module__}.{type(lm).__name__}")
print(f"vision_tower   : {type(vt).__module__}.{type(vt).__name__}")
lm_patched = type(lm).__module__.startswith("openpi.")
vt_patched = type(vt).__module__.startswith("openpi.")
print(f"language_model is patched: {lm_patched}")
print(f"vision_tower  is patched: {vt_patched}")
norm = lm.layers[0].input_layernorm
print(f"AdaRMS live (dense present): {getattr(norm, 'dense', None) is not None}")
assert lm_patched and vt_patched, "submodels resolved to STOCK transformers classes"
assert getattr(norm, "dense", None) is not None, "AdaRMS not configured"
print("\nPALIGEMMA WIRING CORRECT")
