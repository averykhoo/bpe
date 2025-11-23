"""
Hyper-Optimized Trie-based BPE.

OPTIMIZATIONS:
1. @lru_cache on `_is_valid_merge`: This is the biggest speedup. It prevents
   re-calculating the merge rules for common pairs (e.g., " th" + "e") which
   repeats thousands of times.
2. List-based DP: Uses a pre-allocated list [-1] instead of a dictionary for
   `last_token` to avoid hash overhead.
3. Fast-Fail logic: Optimized inner loops to minimize interpreter instruction count.
"""
import base64
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

# Python 3.10+ slots optimization
@dataclass(slots=True)
class TrieNode:
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    token_id: Optional[int] = None


class BytePairEncoding:
    def __init__(self, tokens: List[bytes]):
        self.all_tokens = tokens
        self.token_map = {b: i for i, b in enumerate(tokens)}

        # Build Trie
        self.root = TrieNode()
        for i, token in enumerate(tokens):
            node = self.root
            for byte in token:
                if byte not in node.children:
                    node.children[byte] = TrieNode()
                node = node.children[byte]
            node.token_id = i

        self._build_split_table()

    @classmethod
    def from_tiktoken_file(cls, path: str) -> "BytePairEncoding":
        tokens = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                parts = line.split()
                tokens.append(base64.b64decode(parts[0]))
        return cls(tokens)

    def _find_longest_prefix(self, bytes_seq: bytes) -> Optional[int]:
        node = self.root
        last_valid_token = None
        for byte in bytes_seq:
            if byte not in node.children:
                break
            node = node.children[byte]
            if node.token_id is not None:
                last_valid_token = node.token_id
        return last_valid_token

    def _build_split_table(self):
        self.split_table = [(i, i) for i in range(len(self.all_tokens))]
        self.pair_lookup = {}

        for i, token in enumerate(self.all_tokens):
            if len(token) == 1: continue

            curr_prefix_id = self._find_longest_prefix(token[:-1])

            while curr_prefix_id is not None:
                prefix_len = len(self.all_tokens[curr_prefix_id])
                suffix = token[prefix_len:]

                # Check directly in token_map for speed
                suffix_id = self.token_map.get(suffix)
                if suffix_id is not None:
                    if curr_prefix_id < i and suffix_id < i:
                        # We use the uncached version here during build
                        if self._is_valid_merge_logic(curr_prefix_id, suffix_id):
                            self.pair_lookup[(curr_prefix_id, suffix_id)] = i
                            self.split_table[i] = (curr_prefix_id, suffix_id)
                            break

                curr_prefix_id = self._find_longest_prefix(token[:prefix_len - 1])

        # Clear the cache after build to free memory, though technically empty yet
        self._is_valid_merge.cache_clear()

    # --- CRITICAL OPTIMIZATION: Cache this function ---
    @lru_cache(maxsize=None)
    def _is_valid_merge(self, t1: int, t2: int) -> bool:
        return self._is_valid_merge_logic(t1, t2)

    def _is_valid_merge_logic(self, t1: int, t2: int) -> bool:
        """
        The raw logic for valid merge checking.
        Separated so we can call it uncached during init if needed,
        but cached during encoding.
        """
        limit = float('inf')
        curr_t1, curr_t2 = t1, t2
        # Localize lookups to avoid 'self.' overhead in loop
        pair_lookup = self.pair_lookup
        split_table = self.split_table

        while True:
            if (curr_t1, curr_t2) in pair_lookup:
                if pair_lookup[(curr_t1, curr_t2)] < limit:
                    return False

            if curr_t1 > curr_t2:
                limit = curr_t1
                # Unpack tuple directly
                _, t1_right = split_table[curr_t1]
                curr_t1 = t1_right

                if curr_t1 == limit:
                    limit = curr_t2 + 1
                    t2_left, _ = split_table[curr_t2]
                    curr_t2 = t2_left
                    if curr_t2 + 1 == limit:
                        return True
                continue

            # else curr_t2 >= curr_t1
            limit = curr_t2 + 1
            t2_left, _ = split_table[curr_t2]

            if t2_left == curr_t2:
                return True

            curr_t2 = t2_left
            if curr_t2 + 1 == limit:
                limit = curr_t1
                _, t1_right = split_table[curr_t1]
                curr_t1 = t1_right
                if curr_t1 == limit:
                    return True

    def decode_tokens(self, tokens: List[int]) -> bytes:
        return b"".join(self.all_tokens[t] for t in tokens)

    @lru_cache(maxsize=2**16)
    def encode(self, text: bytes) -> List[int]:
        if not text: return []
        n = len(text)

        # Optimization: Use List instead of Dict for O(1) integer access
        # -1 represents None/Invalid
        last_token = [-1] * (n + 1)
        last_token[0] = 0  # Dummy value

        # Localize variables to avoid 'self.' lookups inside the hot loop
        root = self.root
        is_valid_merge = self._is_valid_merge

        for i in range(n):
            # If position 'i' is not reachable, skip
            if last_token[i] == -1:
                continue

            # Get the token ID that ends at 'i'
            prev_token = last_token[i]

            node = root
            # Scan forward from i
            for j in range(i, n):
                byte = text[j] # Fast integer access on bytes object

                # Fast Fail: standard dictionary lookup
                if byte not in node.children:
                    break
                node = node.children[byte]

                if node.token_id is not None:
                    next_pos = j + 1
                    new_token = node.token_id

                    # Check merge validity
                    # Because of the cache, this becomes nearly O(1) for common words
                    if i == 0:
                        last_token[next_pos] = new_token
                    else:
                        if is_valid_merge(prev_token, new_token):
                            last_token[next_pos] = new_token

        if last_token[n] == -1:
            return []

        # Backward Pass
        encoded = []
        curr = n
        while curr > 0:
            t = last_token[curr]
            encoded.append(t)
            curr -= len(self.all_tokens[t])

        return encoded[::-1]