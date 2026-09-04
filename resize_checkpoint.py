import torch
import os
from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))
from models.config import ModelConfig
from models.transformer import KitAITransformer
from tokenizer.bpe_tokenizer import BPETokenizer
import numpy as np

ckpt_path = Path('checkpoints/checkpoint_final.pt')
tok_path = Path('demo_data/wikitext_tokenizer_proper.json')
out_path = Path('checkpoints/checkpoint_resized.pt')

print('Loading checkpoint...')
ckpt = torch.load(ckpt_path, map_location='cpu')
print('Checkpoint keys:', list(ckpt.keys()))
config_dict = ckpt.get('config', {})
if isinstance(config_dict, dict):
    # cast numeric strings to proper types
    for k in ('d_model','n_layers','n_heads','n_kv_heads','d_ff','max_seq_len','vocab_size'):
        if k in config_dict and isinstance(config_dict[k], str):
            config_dict[k] = int(config_dict[k])
    for k in ('dropout','norm_eps','rope_theta','init_std','init_mean','learning_rate','min_learning_rate','weight_decay','label_smoothing','gradient_clip_val'):
        if k in config_dict and isinstance(config_dict[k], str):
            config_dict[k] = float(config_dict[k])
    base_cfg = ModelConfig(**config_dict)
else:
    raise RuntimeError('Config not dict')
print('Original vocab size from checkpoint:', base_cfg.vocab_size)

# Load tokenizer
tokenizer = BPETokenizer.load(tok_path)
print('Tokenizer vocab size:', tokenizer.vocab_size)
new_vocab_size = tokenizer.vocab_size

# Update vocab size in config
base_cfg.vocab_size = new_vocab_size
print('New vocab size:', base_cfg.vocab_size)

# Build model with new vocab size
model = KitAITransformer(base_cfg)
print('Model built')

# Load state dict with strict=False to ignore mismatched shapes
state_dict = ckpt['model_state_dict']
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f'Missing keys: {len(missing_keys)}')
print(f'Unexpected keys: {len(unexpected_keys)}')
if len(missing_keys) > 0:
    print('First few missing:', missing_keys[:5])
if len(unexpected_keys) > 0:
    print('First few unexpected:', unexpected_keys[:5])

# Now we need to handle embedding and output_head weight resizing
# Since weight_tying=True, embedding.weight and output_head.weight are the same tensor.
# We'll create a new weight matrix.
with torch.no_grad():
    # Get the existing embedding weight from the loaded state (still shape [5000,512])
    # It's currently in model.state_dict() under the appropriate key.
    # Find the embedding weight key:
    emb_weight_key = None
    for k in model.state_dict().keys():
        if 'embedding.weight' in k:
            emb_weight_key = k
            break
    if emb_weight_key is None:
        # fallback: look for 'embed_tokens.weight' or similar
        for k in model.state_dict().keys():
            if 'weight' in k and 'embed' in k.lower():
                emb_weight_key = k
                break
    print(f'Using embedding weight key: {emb_weight_key}')
    old_emb_weight = model.state_dict()[emb_weight_key]  # shape [5000,512]
    # Create new weight matrix
    new_emb_weight = torch.empty(new_vocab_size, old_emb_weight.size(1), dtype=old_emb_weight.dtype, device=old_emb_weight.device)
    # Copy the first 5000 rows
    new_emb_weight[:5000] = old_emb_weight
    # Initialize the rest using same initialization as embedding layer
    # In TokenEmbedding, they likely used nn.Embedding with default init (uniform?).
    # Let's check the model code quickly: but we can approximate with normal(0, init_std)
    init_std = getattr(base_cfg, 'init_std', 0.02)
    torch.nn.init.normal_(new_emb_weight[5000:], mean=0.0, std=init_std)
    # Assign back
    model.state_dict()[emb_weight_key].copy_(new_emb_weight)
    # If weight tying, output_head.weight should point to same tensor; but we just replaced embedding weight.
    # However, if output_head is a separate parameter, we need to copy as well.
    # Let's find output_head weight key:
    out_weight_key = None
    for k in model.state_dict().keys():
        if 'output_head' in k and 'weight' in k:
            out_weight_key = k
            break
    if out_weight_key is not None:
        if out_weight_key == emb_weight_key:
            # already same tensor due to weight tying, done
            pass
        else:
            # copy same new weights
            model.state_dict()[out_weight_key].copy_(new_emb_weight)
    else:
        # If not found, maybe they tie by setting output_head.weight = embedding.weight in forward,
        # so we don't need to set separately.
        pass

print('Embedding and output_head weights resized.')

# Now we need to ensure the state_dict reflects the updated weights.
# Let's rebuild state_dict from model.
new_state_dict = model.state_dict()

# Optionally, we can also keep optimizer and scheduler states from checkpoint if we want to resume training.
# For inference, we only need model_state_dict and config.
new_ckpt = {
    'model_state_dict': new_state_dict,
    'config': {k: v for k, v in base_cfg.__dict__.items() if not k.startswith('_')},
    # we could also keep optimizer_state_dict, scheduler_state, etc. but they'd mismatch shapes; we drop them.
}
torch.save(new_ckpt, out_path)
print(f'Saved resized checkpoint to {out_path}')
print('Done.')
