import sys, time, json
sys.path.insert(0, '.')
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

t0 = time.time()
tok = Tokenizer(BPE(unk_token=None))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
tok.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=['<pad>', '<unk>', '<bos>', '<eos>', '<mask>'],
    show_progress=True,
)
files = [
    'demo_data/wikitext103/train.txt',
    'demo_data/wikitext103/val.txt',
]
tok.train(files, trainer)
out = 'tokenizer/kitai_bpe_32k.json'
tok.save(out)
print(f'trained in {time.time()-t0:.1f}s -> {out}')

# Verify round-trips
tests = [
    'The city of Boston is the capital of Massachusetts.',
    'In 1969, Apollo 11 landed on the Moon.',
    'def f(x):\n    return x * 2\n',
    'Numbers: 3.14159 and 1,000,000.',
    'Unicode: café naïve über — “quotes” …',
]
tot_c = tot_t = 0
for s in tests:
    ids = tok.encode(s).ids
    dec = tok.decode(ids)
    ok = dec == s
    tot_c += len(s); tot_t += len(ids)
    print(('OK  ' if ok else 'FAIL'), repr(dec if not ok else s), f'({len(ids)} toks)')
print(f'tokens/char: {tot_t/tot_c:.3f}')
print('special ids:', {t: tok.token_to_id(t) for t in ['<pad>','<unk>','<bos>','<eos>','<mask>']})
