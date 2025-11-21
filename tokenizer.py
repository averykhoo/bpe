"""
This file corresponds to crates/bpe-openai/src/lib.rs and handles the regex pre-tokenization and normalization.

This file handles the regex pre-tokenization and normalization.
It wraps the core BytePairEncoding logic to provide a high-level API similar to tiktoken.
"""

from typing import List

import regex as re
import unicodedata

from bpe import BytePairEncoding

# Regex patterns matching OpenAI's Rust implementation
# CL100K (GPT-4, gpt-3.5-turbo)
PAT_CL100K = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

# O200K (GPT-4o)
PAT_O200K = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, bpe: BytePairEncoding, pattern: str, nfc_normalize: bool = False):
        self.bpe = bpe
        self.nfc = nfc_normalize
        self.pattern = re.compile(pattern)

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        This corresponds to `encode_ordinary` in tiktoken; it treats special tokens
        (like <|endoftext|>) as regular text, not as control tokens.
        """
        if self.nfc:
            text = unicodedata.normalize("NFC", text)

        tokens = []
        # OpenAI uses a regex to split text into chunks (words/punctuation)
        # BEFORE applying BPE to each chunk.
        for chunk in self.pattern.findall(text):
            chunk_bytes = chunk.encode('utf-8')
            tokens.extend(self.bpe.encode(chunk_bytes))

        return tokens

    def count_tokens(self, text: str) -> int:
        """
        Efficiently counts the number of tokens in a string.

        This is faster and more memory efficient than calling len(encode(text))
        because it does not allocate the full list of integers.
        """
        if self.nfc:
            text = unicodedata.normalize("NFC", text)

        count = 0
        for chunk in self.pattern.findall(text):
            chunk_bytes = chunk.encode('utf-8')
            # BPE.encode returns a list; calculating length is faster than extending a list
            # Optimization Note: Ideally BPE.encode would have a 'count_only' mode,
            # but list allocation in Python is relatively cheap for small regex chunks.
            count += len(self.bpe.encode(chunk_bytes))

        return count

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        Handles invalid UTF-8 sequences gracefully using 'replace'.
        """
        byte_data = self.bpe.decode_tokens(tokens)
        return byte_data.decode('utf-8', errors='replace')


# --- Factories ---

def get_cl100k_base(path: str) -> Tokenizer:
    """
    Returns a tokenizer for the 'cl100k_base' encoding (GPT-4, GPT-3.5-Turbo).
    Requires the path to 'cl100k_base.tiktoken'.
    """
    return Tokenizer(BytePairEncoding.from_tiktoken_file(path), PAT_CL100K, nfc_normalize=False)


def get_o200k_base(path: str) -> Tokenizer:
    """
    Returns a tokenizer for the 'o200k_base' encoding (GPT-4o).
    Requires the path to 'o200k_base.tiktoken'.
    """
    return Tokenizer(BytePairEncoding.from_tiktoken_file(path), PAT_O200K, nfc_normalize=False)
