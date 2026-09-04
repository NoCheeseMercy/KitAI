import torch
import os
from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))
from models.config import ModelConfig
from models.transformer import KitAITransformer
from tokenizer.bpe_tokenizer import BPETokenizer

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

# Prepare state dict from checkpoint, but remove embedding and output_head weights
state_dict = ckpt['model_state_dict']
# Identify keys to exclude
keys_to_remove = []
for k in state_dict.keys():
    if 'embedding.weight' in k or 'output_head.weight' in k:
        keys_to_remove.append(k)
print(f'Removing {len(keys_to_remove)} keys due to size mismatch: {keys_to_remove}')
for k in keys_to_remove:
    del state_dict[k]

# Load the remaining state dict
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f'After loading: missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}')
if len(missing_keys) > 0:
    print('First few missing:', missing_keys[:5])
if len(unexpected_keys) > 0:
    print('First few unexpected:', unexpected_keys[:5])

# Now we need to initialize embedding and output_head weights appropriately.
# Since weight_tying is True in config, embedding.weight and output_head.weight refer to the same tensor.
# We'll initialize the embedding weight with the same scheme as the original embedding.
# Let's find the embedding weight key in the model.
emb_weight_key = None
for k, v in model.named_parameters():
    if 'embedding.weight' in k:
        emb_weight_key = k
        break
if emb_weight_key is None:
    # fallback: look for any parameter with 'weight' and 'embed' in name
    for k, v in model.named_parameters():
        if 'weight' in k and 'embed' in k.lower():
            emb_weight_key = k
            break
print(f'Embedding weight key: {emb_weight_key}')

with torch.no_grad():
    if emb_weight_key is not None:
        # Get the shape
        embed_shape = model.state_dict()[emb_weight_key].shape
        print(f'Embedding shape: {embed_shape}')
        # Initialize with normal distribution using init_std from config
        init_std = getattr(base_cfg, 'init_std', 0.02)
        torch.nn.init.normal_(model.state_dict()[emb_weight_key], mean=0.0, std=init_std)
        # If weight_tying is True, output_head should share the same tensor.
        # Let's verify if there is a separate output_head weight parameter.
        # We'll copy the same data to any output_head weight parameter.
        for name, param in model.named_parameters():
            if 'output_head' in name and 'weight' in name:
                print(f'Copying embedding weights to {name}')
                param.data.copy_(model.state_dict()[emb_weight_key])
    else:
        print('WARNING: Could not find embedding weight to initialize.')

# Now save the new checkpoint
new_ckpt = {
    'model_state_dict': model.state_dict(),
    'config': {k: v for k, v in base_cfg.__dict__.items() if not k.startswith('_')},
}
torch.save(new_ckpt, out_path)
print(f'Saved resized checkpoint to {out_path}')
print('Done.')
