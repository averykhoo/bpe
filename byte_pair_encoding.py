"""
This file contains the direct port of crates/bpe/src/byte_pair_encoding.rs

Core Byte Pair Encoding (BPE) implementation.

This is a pure Python port of the "Novel Algorithm" described in the `bpe` Rust crate.
Unlike traditional BPE (which greedily merges pairs), this implementation:
1. Reverse-engineers the merge tree at initialization.
2. Uses Dynamic Programming to find the correct tokenization in O(N) time.
3. Ensures exact compatibility with OpenAI's tiktoken.
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
        """
        Initialize BPE from a list of byte tokens.

        Args:
            tokens: A list of bytes, where the index corresponds to the token ID (rank).
                    The list must be sorted by rank (0 to vocab_size).
        """
        self.all_tokens = tokens
        # Direct map for O(1) exact lookups
        self.token_map = {b: i for i, b in enumerate(tokens)}

        # Build the Trie for efficient prefix matching
        self.root = TrieNode()
        for i, token in enumerate(tokens):
            node = self.root
            for byte in token:
                if byte not in node.children:
                    node.children[byte] = TrieNode()
                node = node.children[byte]
            node.token_id = i

        # Precompute split table for valid merge checks.
        # This is computationally expensive, so it is handled in a separate method
        # that can be cached.
        self._build_split_table()

    @classmethod
    def from_tiktoken_file(cls, path: str) -> "BytePairEncoding":
        tokens = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                parts = line.split()
                # File format: base64_token rank
                tokens.append(base64.b64decode(parts[0]))
        return cls(tokens)

    def _find_longest_prefix(self, bytes_seq: bytes) -> Optional[int]:
        """Finds the token_id of the longest prefix of bytes_seq that matches a token."""
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
        This determines which two sub-tokens (t1, t2) were merged to create a specific token.
        This is crucial for the O(N) encoding algorithm.
        """
        # Initialize split_table with (i, i) indicating base tokens (not merged)
        self.split_table = [(i, i) for i in range(len(self.all_tokens))]
        self.pair_lookup = {}

        for i, token in enumerate(self.all_tokens):
            if len(token) == 1: continue

            # Find the split point
            # 1. Start with the longest possible prefix that is a token
            curr_prefix_id = self._find_longest_prefix(token[:-1])

            # 2. Iterate backwards (trying shorter prefixes) until we find the valid BPE split
            found = False
            while curr_prefix_id is not None:
                prefix_len = len(self.all_tokens[curr_prefix_id])
                suffix = token[prefix_len:]

                if suffix in self.token_map:
                    suffix_id = self.token_map[suffix]

                    # BPE Rule: The components (prefix, suffix) must have existed *before*
                    # the merged token was created. This means their ranks (IDs) must be lower.
                    if curr_prefix_id < i and suffix_id < i:
                        # Check if this merge is valid according to the greedy history
                        if self._is_valid_merge(curr_prefix_id, suffix_id):
                            self.pair_lookup[(curr_prefix_id, suffix_id)] = i
                            self.split_table[i] = (curr_prefix_id, suffix_id)
                            found = True
                            break

                # Backtrack: try the next longest prefix
                curr_prefix_id = self._find_longest_prefix(token[:prefix_len - 1])

    def _is_valid_merge(self, t1: int, t2: int) -> bool:
        """
        Checks if merging t1 and t2 is consistent with BPE rules.

        This verifies that the greedy BPE algorithm wouldn't have merged
        parts of t1 or t2 with each other *before* merging t1 and t2.
        """
        limit = float('inf')
        curr_t1, curr_t2 = t1, t2

        while True:
            # If (curr_t1, curr_t2) forms a known token with a lower rank than our current limit,
            # then BPE would have merged them *earlier*. This makes the proposed merge invalid.
            if (curr_t1, curr_t2) in self.pair_lookup:
                if self.pair_lookup[(curr_t1, curr_t2)] < limit:
                    return False

            # We need to descend the merge tree to check boundaries.
            if curr_t1 > curr_t2:
                # Descend into T1.
                # Since T1 is on the Left, and we check the boundary (T1, T2),
                # we must look at the RIGHT child of T1.
                limit = curr_t1
                t1_left, t1_right = self.split_table[curr_t1]

                # We descend into the RIGHT child of T1 because we are checking the
                # boundary between (T1, T2).
                curr_t1 = t1_right

                if curr_t1 == limit:
                    # T1 is a base token (split into itself).
                    # We can't descend T1 further, so we switch to descending T2.
                    limit = curr_t2 + 1
                    t2_left, t2_right = self.split_table[curr_t2]
                    curr_t2 = t2_left  # Left child of T2 touches the boundary
                    if curr_t2 + 1 == limit:
                        return True
                continue

            # Else (curr_t2 >= curr_t1):
            # T2 is newer. Descend into T2.
            # Since T2 is on the Right, and we check the boundary (T1, T2),
            # we must look at the LEFT child of T2.
            limit = curr_t2 + 1
            t2_left, t2_right = self.split_table[curr_t2]

            if t2_left == curr_t2:
                # T2 is a base token.
                return True

            curr_t2 = t2_left  # Descend Left child of T2 (touching boundary)
            if curr_t2 + 1 == limit:
                # Switch back to descending T1
                limit = curr_t1
                t1_left, t1_right = self.split_table[curr_t1]
                curr_t1 = t1_right
                if curr_t1 == limit:
                    return True

    def decode_tokens(self, tokens: List[int]) -> bytes:
        """Decodes a list of token IDs back to bytes."""
        return b"".join(self.all_tokens[t] for t in tokens)

    def encode(self, text: bytes) -> List[int]:
        """
        Encodes text using the 'Novel Algorithm' (Dynamic Programming).

        Instead of repeatedly scanning for the best pair (O(N^2)), we scan
        forward to find all valid token endpoints, then backtrack to find the
        optimal path. This runs in O(N).
        """
        if not text: return []
        n = len(text)

        # last_token[i] stores the token_id that *correctly* ends at position i.
        # We initialize index 0 with None because it is the valid start position,
        # effectively using the dict keys as the set of reachable positions.
        last_token: Dict[int, Optional[int]] = {0: None}

        # 1. Forward Pass: Find valid tokens ending at every position
        for i in range(n):
            # If i is not in last_token, it means no valid sequence of tokens
            # reaches this position, so we cannot start a new token here.
            if i not in last_token: continue

            node = self.root
            for j in range(i, n):
                byte = text[j]
                if byte not in node.children: break
                node = node.children[byte]

                if node.token_id is not None:
                    next_pos = j + 1
                    new_token = node.token_id

                    # A token is valid here if:
                    # a) It's at the very start of the string (i==0)
                    # b) It merges validly with the token preceding it
                    is_valid = False
                    if i == 0:
                        is_valid = True
                    else:
                        # We look up the token that ended at 'i' to check compatibility.
                        # Since i > 0, last_token[i] is guaranteed to be an int (not None).
                        prev_token = last_token[i]
                        if self._is_valid_merge(prev_token, new_token):
                            is_valid = True

                    if is_valid:
                        # We found a valid path to 'next_pos'.
                        # In this DP approach, we simply overwrite; Corollary IIa
                        # from the Rust implementation guarantees uniqueness.
                        last_token[next_pos] = new_token

        # 2. Backward Pass: Reconstruct the sequence
        if n not in last_token:
            # If we couldn't reach the end, the byte sequence is likely invalid
            # for this tokenizer (e.g., partial UTF-8 bytes or invalid Unicode).
            return []

        encoded = []
        curr = n
        while curr > 0:
            t = last_token[curr]
            encoded.append(t)
            # To backtrack, we subtract the length of the current token
            curr -= len(self.all_tokens[t])

        return encoded[::-1]
