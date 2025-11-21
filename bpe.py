"""
This file contains the direct port of crates/bpe/src/byte_pair_encoding.rs
"""
import base64
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional


# Python 3.10+ supports slots=True, which drastically reduces RAM usage.
# If you are on an older version, remove (slots=True) but be warned:
# RAM usage will double for large vocabs like cl100k.
@dataclass(slots=True)
class TrieNode:
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    token_id: Optional[int] = None


class BytePairEncoding:
    def __init__(self, tokens: List[bytes]):
        self.all_tokens = tokens
        # Direct map for O(1) lookups
        self.token_map = {b: i for i, b in enumerate(tokens)}

        # Build the Trie
        self.root = TrieNode()
        for i, token in enumerate(tokens):
            node = self.root
            for byte in token:
                if byte not in node.children:
                    node.children[byte] = TrieNode()
                node = node.children[byte]
            node.token_id = i

        # Precompute split table for valid merge checks
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
        """Finds the longest prefix of bytes_seq that is a valid token."""
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
        """
        Reverse-engineers the BPE merge tree.
        Determines which two tokens merge to form a larger token.
        """
        self.split_table = [(i, i) for i in range(len(self.all_tokens))]
        self.pair_lookup = {}

        for i, token in enumerate(self.all_tokens):
            if len(token) == 1: continue

            # Find the split point
            # 1. Start with the longest prefix
            curr_prefix_id = self._find_longest_prefix(token[:-1])

            # 2. Iterate backwards to find the specific merge that created this token
            found = False
            while curr_prefix_id is not None:
                prefix_len = len(self.all_tokens[curr_prefix_id])
                suffix = token[prefix_len:]

                if suffix in self.token_map:
                    suffix_id = self.token_map[suffix]

                    # Merge rules: components must exist before the result (lower rank/index)
                    if curr_prefix_id < i and suffix_id < i:
                        if self._is_valid_merge(curr_prefix_id, suffix_id):
                            self.pair_lookup[(curr_prefix_id, suffix_id)] = i
                            self.split_table[i] = (curr_prefix_id, suffix_id)
                            found = True
                            break

                # Try next shorter prefix
                curr_prefix_id = self._find_longest_prefix(token[:prefix_len - 1])

    def _is_valid_merge(self, t1: int, t2: int) -> bool:
        """
        The core validation logic from the Rust implementation.
        Ensures that merging t1 and t2 follows the greedy BPE rules established
        by previous tokens.
        """
        limit = float('inf')
        curr_t1, curr_t2 = t1, t2

        while True:
            # If these two tokens form a known higher-priority token,
            # BPE would have eagerly merged them earlier.
            if (curr_t1, curr_t2) in self.pair_lookup:
                if self.pair_lookup[(curr_t1, curr_t2)] < limit:
                    return False

            # Traverse down the merge tree
            if curr_t1 > curr_t2:
                # Descend into T1.
                # Since T1 is on the Left, and we check the boundary (T1, T2),
                # we must look at the RIGHT child of T1.
                limit = curr_t1
                t1_left, t1_right = self.split_table[curr_t1]
                curr_t1 = t1_right

                if curr_t1 == limit:
                    # T1 was a base token (split into itself).
                    # We can't descend T1, so we switch to descending T2.
                    limit = curr_t2 + 1
                    t2_left, t2_right = self.split_table[curr_t2]
                    curr_t2 = t2_left
                    if curr_t2 + 1 == limit:
                        return True
                continue

            # Else (curr_t2 >= curr_t1):
            # Descend into T2.
            # Since T2 is on the Right, and we check the boundary (T1, T2),
            # we must look at the LEFT child of T2.
            limit = curr_t2 + 1
            t2_left, t2_right = self.split_table[curr_t2]

            if t2_left == curr_t2:
                # T2 was a base token.
                return True

            curr_t2 = t2_left
            if curr_t2 + 1 == limit:
                # Switch back to T1
                limit = curr_t1
                t1_left, t1_right = self.split_table[curr_t1]
                curr_t1 = t1_right
                if curr_t1 == limit:
                    return True

    def decode_tokens(self, tokens: List[int]) -> bytes:
        return b"".join(self.all_tokens[t] for t in tokens)

    def encode(self, text: bytes) -> List[int]:
        """
        The 'Novel Algorithm' (Dynamic Programming).
        Finds the correct sequence of tokens using the validity check.
        """
        if not text: return []
        n = len(text)

        # last_token[i] stores the token_id that correctly ends at position i
        last_token: Dict[int, int] = {}
        valid_indices = {0}

        # Forward Pass
        for i in range(n):
            if i not in valid_indices: continue

            node = self.root
            for j in range(i, n):
                byte = text[j]
                if byte not in node.children: break
                node = node.children[byte]

                if node.token_id is not None:
                    next_pos = j + 1
                    new_token = node.token_id

                    is_valid = False
                    if i == 0:
                        is_valid = True
                    else:
                        prev_token = last_token[i]
                        if self._is_valid_merge(prev_token, new_token):
                            is_valid = True

                    if is_valid:
                        last_token[j + 1] = new_token
                        valid_indices.add(j + 1)

        # Backward Pass (Reconstruction)
        if n not in last_token:
            # Fallback for byte sequences that don't map perfectly (rare in cl100k,
            # but possible with bad utf8/partial bytes).
            return []

        encoded = []
        curr = n
        while curr > 0:
            t = last_token[curr]
            encoded.append(t)
            curr -= len(self.all_tokens[t])

        return encoded[::-1]
