import sys, math, torch
sys.path.insert(0, '.')
ckpt = torch.load('checkpoints/overnight_cache_20260817/checkpoint_step_00002000.pt',
                 map_location='cpu', weights_only=False)
print('checkpoint keys:', list(ckpt.keys()))
print('stored step:', ckpt.get('step'), 'epoch:', ckpt.get('epoch'))

sch = ckpt.get('scheduler_state')
print('\n--- scheduler_state ---')
for k, v in sch.items():
    if isinstance(v, (int, float, str, list)):
        print(f'  {k}: {v!r}')

opt = ckpt.get('optimizer_state_dict')
if opt:
    pgs = opt.get('param_groups')
    if pgs:
        print('\n--- optimizer param_groups (first group) ---')
        pg0 = pgs[0]
        print('  lr:', pg0.get('lr'))
        print('  initial_lr:', pg0.get('initial_lr'))
        print('  betas:', pg0.get('betas'), 'wd:', pg0.get('weight_decay'))

# Reproduce expected LR at step with FULL 25000 schedule
warmup, total, max_lr, min_lr = 1000, 25000, 5e-4, 5e-5
def lr_at(step):
    if step < warmup:
        return max_lr * (step / max(1, warmup))
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(max(prog, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return min_lr + (max_lr - min_lr) * cos

print('\n--- expected full-schedule LR ---')
for s in [0, 500, 1000, 2000, 5000, 12500, 24000, 24999]:
    print(f'  step {s:6d}: {lr_at(s):.6e}')
