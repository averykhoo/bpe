"""
OpenAI Tokenizer Wrapper

This file corresponds to crates/bpe-openai/src/lib.rs

This module handles the pre-tokenization logic (Regex splitting) and normalization (NFC)
required to replicate OpenAI's GPT-4 and GPT-4o tokenizers. It wraps the core
BytePairEncoding logic to provide a high-level API similar to `tiktoken`.
"""

from itertools import chain
from pathlib import Path
from typing import List
from typing import Pattern
from typing import Union

import regex
import unicodedata

from byte_pair_encoding import BytePairEncoding

# --- Pre-compiled Regex Patterns ---

# CL100K (GPT-4, gpt-3.5-turbo)
# Logic: Contractions -> Words -> Numbers -> Punctuation -> Whitespace
# Note: This uses standard regex logic without recursion/definitions.
CL100K_PATTERN = regex.compile(r"""
    (?i:'s|'t|'re|'ve|'m|'ll|'d)|   # Contractions (case-insensitive)
    [^\r\n\p{L}\p{N}]?\p{L}+|       # Words (optional prefix non-letter/num)
    \p{N}{1,3}|                     # Numbers (1 to 3 digits)
    \ ?[^\s\p{L}\p{N}]+[\r\n]*|     # Punctuation / symbols
    \s*[\r\n]+|                     # Newlines (and surrounding whitespace)
    \s+(?!\S)|                      # Trailing whitespace
    \s+                             # Other whitespace
""", regex.VERBOSE)

# OPTIMIZED O200K PATTERN (Flattened)
# We expanded the (?&define) groups directly into the main pattern.
# This removes the overhead of PCRE subroutine calls during matching.
O200K_PATTERN = regex.compile(r"""
    # 1. Mixed/Lowercase words: prefix + upper* + lower+ + suffix
    # \p{Lu}: Uppercase, \p{Lt}: Titlecase, \p{Lm}: Modifier, \p{Lo}: Other, \p{M}: Mark
    # \p{Ll}: Lowercase
    [^\r\n\p{L}\p{N}]?                                      # Prefix
    [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*                        # Upper*
    [\p{Ll}\p{Lm}\p{Lo}\p{M}]+                              # Lower+
    (?i:'s|'t|'re|'ve|'m|'ll|'d)?|                          # Suffix

    # 2. Uppercase words: prefix + upper+ + lower* + suffix
    [^\r\n\p{L}\p{N}]?                                      # Prefix
    [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+                        # Upper+
    [\p{Ll}\p{Lm}\p{Lo}\p{M}]*                              # Lower*
    (?i:'s|'t|'re|'ve|'m|'ll|'d)?|                          # Suffix

    # 3. Numbers
    \p{N}{1,3}|

    # 4. Punctuation
    \ ?[^\s\p{L}\p{N}]+[\r\n/]*|

    # 5. Newlines
    \s*[\r\n]+|

    # 6. Whitespace
    \s+(?!\S)|
    \s+
""", regex.VERBOSE)


class Tokenizer:
    """
    A high-level tokenizer that splits text via Regex before applying BPE.
    """

    def __init__(self, bpe: BytePairEncoding, pattern: Pattern, nfc_normalize: bool = False):
        """
        Initialize the tokenizer.

        Args:
            bpe: The loaded BytePairEncoding instance (dictionary logic).
            pattern: The compiled regex pattern used to split text into chunks.
            nfc_normalize: If True, normalize text to Unicode NFC before processing.
                           (Default: False, to match strict tiktoken behavior).
        """
        self.bpe = bpe
        self.nfc = nfc_normalize
        self.pattern = pattern

    def __call__(self, text: str) -> List[int]:
        """
        Allows usage as a callable: `tokens = tokenizer("my text")`.
        """
        return self.encode(text)

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        This treats special tokens (like <|endoftext|>) as regular text.
        To handle special tokens, they must be split out before calling this method.

        Args:
            text: The input string.

        Returns:
            A list of integers representing the tokens.
        """
        if self.nfc:
            text = unicodedata.normalize("NFC", text)

        # Optimization: Local variable for speed
        encode_func = self.bpe.encode_chunk_str

        # Optimization: List Comprehension + Chain
        # 1. findall returns a list of strings.
        # 2. [encode_func(s) ...] creates a list of references to cached integer lists.
        # 3. chain.from_iterable flattens this list-of-lists into a single iterator.
        # 4. list(...) consumes it at C-speed.
        return list(chain.from_iterable(
            [encode_func(chunk) for chunk in self.pattern.findall(text)]
        ))

    def count_tokens(self, text: str) -> int:
        """
        Efficiently counts the number of tokens in a string.

        This is more memory efficient than `len(encode(text))` because it avoids
        allocating the full list of integers, only summing the lengths.

        Args:
            text: The input string.

        Returns:
            The total number of tokens.
        """
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        encode_func = self.bpe.encode_chunk_str
        # Generator expression is memory efficient here
        return sum(len(encode_func(chunk)) for chunk in self.pattern.findall(text))

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            tokens: List of token integers.

        Returns:
            The decoded string. Invalid UTF-8 sequences are replaced with the
            Unicode replacement character.
        """
        byte_data = self.bpe.decode_tokens(tokens)
        return byte_data.decode('utf-8', errors='replace')


# --- Factories ---

def get_cl100k_base(path: Union[str, Path]) -> Tokenizer:
    """
    Factory for the 'cl100k_base' tokenizer (GPT-4, GPT-3.5-Turbo, text-embedding-3).

    Args:
        path: Path to the `cl100k_base.tiktoken` file.

    Returns:
        A configured Tokenizer instance.
    """
    return Tokenizer(BytePairEncoding.from_tiktoken_file(path), CL100K_PATTERN)


def get_o200k_base(path: Union[str, Path]) -> Tokenizer:
    """
    Factory for the 'o200k_base' tokenizer (GPT-4o).

    Args:
        path: Path to the `o200k_base.tiktoken` file.

    Returns:
        A configured Tokenizer instance.
    """
    return Tokenizer(BytePairEncoding.from_tiktoken_file(path), O200K_PATTERN)
