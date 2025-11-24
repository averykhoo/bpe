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

# ==============================================================================
# CL100K PATTERN (GPT-4, GPT-3.5-Turbo, text-embedding-3)
# ==============================================================================
# This regex splits text based on categories: contractions, words, numbers,
# punctuation, and whitespace. It is relatively flat and O(N) efficient.
CL100K_PATTERN = regex.compile(r"""
    # 1. Contractions (Case-Insensitive)
    # Matches: 's, 't, 're, 've, 'm, 'll, 'd
    # The (?i:...) group creates a case-insensitive block.
    (?i:'s|'t|'re|'ve|'m|'ll|'d)|

    # 2. Words
    # Matches: "word", " word", "\nword" (if following a newline)
    # Logic: Optional single non-alphanumeric prefix, followed by 1+ letters.
    [^\r\n\p{L}\p{N}]?\p{L}+|

    # 3. Numbers
    # Matches: "1", "12", "123"
    # Logic: 1 to 3 numeric digits. Longer numbers are split (e.g. 1000 -> 100, 0).
    \p{N}{1,3}|

    # 4. Punctuation / Symbols
    # Matches: "!", " !", "...", "--------"
    # Logic: Optional space, followed by 1+ non-whitespace/non-alphanumeric chars,
    # followed by optional newlines.
    \ ?[^\s\p{L}\p{N}]+[\r\n]*|

    # 5. Newlines
    # Matches: "\n", "\n\n", "  \n"
    # Logic: Optional whitespace followed by 1+ newline characters.
    \s*[\r\n]+|

    # 6. Trailing Whitespace
    # Matches: "   " (at the very end of the string)
    # Logic: Whitespace that is NOT followed by a non-whitespace character.
    \s+(?!\S)|

    # 7. Other Whitespace
    # Matches: " ", "  "
    \s+
""", regex.VERBOSE)

# ==============================================================================
# O200K PATTERN (GPT-4o) - OPTIMIZED "MANUAL DFA"
# ==============================================================================
# BACKGROUND:
# The original OpenAI definition contains two overlapping rules:
#   Rule 1: [Prefix] [Upper]* [Lower]+ [Suffix]  (Matches "Hello", "hello")
#   Rule 2: [Prefix] [Upper]+ [Lower]* [Suffix]  (Matches "Hello", "HELLO")
#
# PROBLEM:
# These rules cause severe performance issues in Python's regex engine due to backtracking.
# If the input is "HELLO" (Uppercase), the engine tries Rule 1 first, consumes "HELLO",
# looks for a lowercase letter, fails, BACKTRACKS to the start, and tries Rule 2.
#
# OPTIMIZATION:
# We refactored these into two MUTUALLY EXCLUSIVE branches. This acts like a
# Deterministic Finite Automaton (DFA), allowing the engine to pick the correct path
# based solely on the first character.
#
#   Branch A: Starts with a Lowercase-ish char. (Covers Rule 1's "hello" case)
#   Branch B: Starts with an Uppercase-ish char. (Covers Rule 2 + Rule 1's "Hello" case)
#
# UNICODE CLASSES:
# \p{Lu}: Uppercase Letter    \p{Ll}: Lowercase Letter
# \p{Lt}: Titlecase Letter    \p{Lm}: Modifier Letter
# \p{Lo}: Other Letter        \p{M}:  Mark (Accents, etc.)

O200K_PATTERN = regex.compile(r"""
    # 1. Common Prefix
    # Matches: Optional single character that is NOT a newline, letter, or number.
    [^\r\n\p{L}\p{N}]?
    
    # 2. The Mutually Exclusive Core (The Optimization)
    (?:
        # Branch A: Words starting with Lowercase/Modifier/Other
        # Matches: "hello", "snake_case", "éléphant"
        # Logic: Must contain at least one of these chars.
        [\p{Ll}\p{Lm}\p{Lo}\p{M}]+
        |
        # Branch B: Words starting with Uppercase/Titlecase
        # Matches: "Hello", "HELLO", "CamelCase"
        # Logic: 1+ Uppercase chars, followed by 0+ Lowercase chars.
        [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+ [\p{Ll}\p{Lm}\p{Lo}\p{M}]*
    )
    
    # 3. Common Suffix (Contractions)
    # Matches: 's, 't, 're, ... (Case-Insensitive)
    (?i:'s|'t|'re|'ve|'m|'ll|'d)?
    
    # --- The following are standard alternatives ---

    # 4. Numbers
    # Matches: 1 to 3 digits.
    | \p{N}{1,3}
    
    # 5. Punctuation
    # Matches: Symbols, including forward slash '/'.
    | \ ?[^\s\p{L}\p{N}]+[\r\n/]*
    
    # 6. Newlines (optimizedP to prevent backtracking
    | [^\S\r\n]*[\r\n]+
    
    ## 7. Trailing Whitespace
    #| \s+(?!\S) # removed because it duplicates the check below and just backtracks
    
    # 8. Other Whitespace
    | \s+
""", regex.VERBOSE)


class Tokenizer:
    """
    A high-level tokenizer that splits text via Regex before applying BPE.
    """

    def __init__(self, bpe: BytePairEncoding, pattern: Pattern, nfc_normalize: bool = False):
        """
        Initialize the tokenizer.

        Args:
            bpe: The loaded BytePairEncoding instance (must support `encode_chunk_str`).
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

        PERFORMANCE NOTES:
        1. We use `itertools.chain.from_iterable` to flatten the list of lists.
           This is faster than repeatedly calling `list.extend` in Python loops.
        2. We use `bpe.encode_chunk_str` (the cached method). This ensures that
           common words (like " the") are not repeatedly re-encoded or re-converted
           to UTF-8 bytes.

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
