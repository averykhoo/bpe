"""
This file corresponds to crates/bpe-openai/src/lib.rs and handles the regex pre-tokenization and normalization
"""
from typing import List

import regex as re
import unicodedata

from bpe import BytePairEncoding

# Regex patterns matching OpenAI's Rust implementation
PAT_CL100K = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
PAT_O200K = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, bpe: BytePairEncoding, pattern: str, nfc_normalize: bool = False):
        self.bpe = bpe
        self.nfc = nfc_normalize
        self.pattern = re.compile(pattern)

    def encode(self, text: str) -> List[int]:
        if self.nfc: text = unicodedata.normalize("NFC", text)
        tokens = []
        for chunk in self.pattern.findall(text):
            tokens.extend(self.bpe.encode(chunk.encode('utf-8')))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.bpe.decode_tokens(tokens).decode('utf-8', errors='replace')


# Factory
def get_cl100k_base(path: str) -> Tokenizer:
    return Tokenizer(BytePairEncoding.from_tiktoken_file(path), PAT_CL100K)
# Factory
def get_o200k_base(path: str) -> Tokenizer:
    return Tokenizer(BytePairEncoding.from_tiktoken_file(path), PAT_O200K)
