from tokenizers import ByteLevelBPETokenizer
import os

print('Files exist:', os.path.exists('wikitext_combined.txt'))
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=['wikitext_combined.txt'], vocab_size=32000, min_frequency=2, special_tokens=[
    '\n',
    '<pad>',
    '\t',
    '
