import sys
sys.path.insert(0, '.')
from tokenizer.bpe_tokenizer import BPETokenizer

tok = BPETokenizer.load('demo_data/wikitext_tokenizer_converted.json')
v = tok.token_to_id
cands = [' the', ' and', ' to', ' of', ' in', ' was', ' said', ' The', ' city', ' Boston']
for c in cands:
    ids = tok.encode(c, add_special_tokens=False)
    toks = [tok.id_to_token[i] for i in ids]
    print(f'{c!r:12} -> {len(ids)} tokens: {toks}')
print()
s = 'The city of Boston is the capital of Massachusetts.'
ids = tok.encode(s, add_special_tokens=False)
print('tokens/char:', len(ids)/len(s))
print('decode:', repr(tok.decode(ids)))
