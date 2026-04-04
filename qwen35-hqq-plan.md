# HQQ Support for Qwen3.5 Family Models

## Architecture Overview

Qwen3.5-9B (`Qwen3_5ForConditionalGeneration`) is a vision-language model.
The module hierarchy is:

```
Qwen3_5ForConditionalGeneration
├── model  (Qwen3_5Model)
│   ├── visual           (Qwen3_5VisionModel)      ← image/video encoder, keep in fp
│   └── language_model   (Qwen3_5TextModel)        ← text decoder, quantize this
│       ├── embed_tokens
│       ├── layers[i]    (Qwen3_5DecoderLayer)
│       │   ├── self_attn  (Qwen3_5Attention)      ← standard layers (most)
│       │   │   ├── q_proj, k_proj, v_proj, o_proj
│       │   │   └── q_norm, k_norm  (QK-RMSNorm)
│       │   ├── linear_attn (Qwen3_5GatedDeltaNet) ← hybrid layers (some)
│       │   ├── mlp
│       │   │   ├── gate_proj, up_proj, down_proj
│       │   ├── input_layernorm
│       │   └── post_attention_layernorm
│       ├── norm
│       └── rotary_emb
└── lm_head
```

## Sample Code: Loading and Running Qwen3.5-9B

```python
import torch
import re
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen3.5-9B"

# Processor handles both text tokenization and image/video preprocessing
processor = AutoProcessor.from_pretrained(model_id)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
model.eval()

# ── Text-only inference ──────────────────────────────────────────────────────
messages = [{"role": "user", "content": "Explain quantum entanglement simply."}]

# Thinking mode (default) — model reasons inside <think>…</think>
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor(text=text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=8192,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.5,
    )

response = processor.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
# Strip reasoning block, keep final answer
answer = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
print(answer)

# ── Non-thinking (instruct) mode ─────────────────────────────────────────────
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = processor(text=text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.5,
    )

print(processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## HQQ Quantization Usage

```python
import torch
from transformers import AutoProcessor
from hqq.engine.hf import HQQModelForCausalLM
from hqq.core.quantize import BaseQuantizeConfig
from hqq.utils.patching import prepare_for_inference

model_id = "Qwen/Qwen3.5-9B"
processor = AutoProcessor.from_pretrained(model_id)

model = HQQModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
model.quantize_model(quant_config=quant_config, compute_dtype=torch.bfloat16, device="cuda")

prepare_for_inference(model, backend="torchao_int4")
model.save_quantized("Qwen3.5-9B-hqq-4bit")
```

## Files to Change

### 1. `hqq/models/hf/qwen35.py` — NEW FILE

Two-class pattern identical to every other HF architecture in the library:

- `Qwen35Patch(BasePatch)` — declares layer layout and patching logic
- `Qwen35HQQ(Qwen35Patch, BaseHQQHFModel)` — concrete registered class

**Key differences vs. Llama/Mistral patch:**

| Aspect             | Llama/Mistral         | Qwen3.5                                        |
|--------------------|-----------------------|------------------------------------------------|
| Text model path    | `model.model`         | `model.model.language_model`                   |
| `lm_head` path     | `model.lm_head`       | `model.lm_head`                                |
| Attention norms    | none                  | `q_norm`, `k_norm` per layer (non-linear)      |
| Layer uniformity   | all same type         | hybrid: check `hasattr(layer, "self_attn")`    |
| Vision encoder     | n/a                   | `model.model.visual` — skip (keep in fp)       |

**`get_linear_tags()`** — standard attention + MLP projections only; `q_norm`/`k_norm`
are RMSNorm (non-linear) and handled in `patch_nonlinearlayers` instead.

**`patch_nonlinearlayers()`** — moves to device: `embed_tokens`, `norm`, `lm_head`,
per-layer `input_layernorm`, `post_attention_layernorm`, `rotary_emb`, `mlp.act_fn`,
and per-layer `self_attn.q_norm` / `self_attn.k_norm` (only on standard-attention layers).

**`patch_linearlayers()`** — iterates `model.model.language_model.layers`; checks
`hasattr(layer, "self_attn")` before accessing attention projections to skip
`Qwen3_5GatedDeltaNet` hybrid layers cleanly.

### 2. `hqq/engine/hf.py` — ONE LINE ADDITION

Register the new class:

```python
from hqq.models.hf.qwen35 import Qwen35HQQ
_HQQ_REGISTRY["Qwen3_5ForConditionalGeneration"] = Qwen35HQQ
```
