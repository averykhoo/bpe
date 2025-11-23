"""
Maximum Performance Pure Python BPE (Flat-Trie + Bitwise Optimization).

CHANGELOG vs Previous:
1. REMOVED 'class TrieNode'. Replaced with parallel lists (State Machine).
   - Speedup: Eliminates object attribute lookups (node.children) completely.
   - Memory: Reduces RAM usage significantly for large vocabs like o200k.
2. BITWISE KEYS: Replaced `(t1, t2)` tuples in `pair_lookup` with `(t1 << 20) | t2`.
   - Speedup: Avoids allocating tuple objects in the hot merge-check loop.
3. INLINED LOOPS: The encode loop now accesses lists directly by integer index.
"""
import base64
from functools import lru_cache
from typing import Dict
from typing import List
from typing import Optional


class BytePairEncoding:
    __slots__ = (
        'all_tokens', 'token_map', 'split_table', 'pair_lookup',
        'trie_children', 'trie_token_ids', 'trie_root'
    )

    def __init__(self, tokens: List[bytes]):
        self.all_tokens = tokens
        self.token_map = {b: i for i, b in enumerate(tokens)}

        # --- FLAT TRIE CONSTRUCTION (Integer State Machine) ---
        # State 0 is root.
        # trie_children[state_id][byte] -> next_state_id
        self.trie_children: List[Dict[int, int]] = [{}]
        # trie_token_ids[state_id] -> token_id (or -1 if none)
        self.trie_token_ids: List[int] = [-1]

        for i, token in enumerate(tokens):
            state = 0
            for byte in token:
                # Direct access to the dictionary for the current state
                children = self.trie_children[state]
                if byte not in children:
                    new_state = len(self.trie_children)
                    self.trie_children.append({})
                    self.trie_token_ids.append(-1)
                    children[byte] = new_state
                    state = new_state
                else:
                    state = children[byte]
            self.trie_token_ids[state] = i

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
        # Optimized to use flat lists
        state = 0
        last_valid_token = None
        children_list = self.trie_children
        token_ids_list = self.trie_token_ids

        for byte in bytes_seq:
            children = children_list[state]
            if byte not in children:
                break
            state = children[byte]
            if token_ids_list[state] != -1:
                last_valid_token = token_ids_list[state]
        return last_valid_token

    def _build_split_table(self):
        self.split_table = [(i, i) for i in range(len(self.all_tokens))]
        # OPTIMIZATION: Use Int keys (t1 << 20 | t2) instead of Tuples
        # Assuming max vocab size < 2^20 (~1 million), which covers o200k easily.
        self.pair_lookup = {}

        for i, token in enumerate(self.all_tokens):
            if len(token) == 1: continue

            curr_prefix_id = self._find_longest_prefix(token[:-1])

            while curr_prefix_id is not None:
                prefix_len = len(self.all_tokens[curr_prefix_id])
                suffix = token[prefix_len:]

                suffix_id = self.token_map.get(suffix)
                if suffix_id is not None:
                    if curr_prefix_id < i and suffix_id < i:
                        # Uncached check during build
                        if self._is_valid_merge_logic(curr_prefix_id, suffix_id):
                            # Store with BITWISE KEY
                            key = (curr_prefix_id << 20) | suffix_id
                            self.pair_lookup[key] = i
                            self.split_table[i] = (curr_prefix_id, suffix_id)
                            break

                curr_prefix_id = self._find_longest_prefix(token[:prefix_len - 1])

        self._is_valid_merge.cache_clear()

    @lru_cache(maxsize=None)
    def _is_valid_merge(self, t1: int, t2: int) -> bool:
        return self._is_valid_merge_logic(t1, t2)

    def _is_valid_merge_logic(self, t1: int, t2: int) -> bool:
        limit = float('inf')
        curr_t1, curr_t2 = t1, t2

        # Localize for speed
        pair_lookup = self.pair_lookup
        split_table = self.split_table

        while True:
            # key = (curr_t1 << 20) | curr_t2  # attempted optimization
            key = (t1, t2)  # turns out tuple allocation is as fast as bitwise math

            if key in pair_lookup:
                if pair_lookup[key] < limit:
                    return False

            if curr_t1 > curr_t2:
                limit = curr_t1
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

    # Cache the chunk encoding (vital for Regex splitter approach)
    @lru_cache(maxsize=2 ** 16)
    def encode(self, text: bytes) -> List[int]:
        if not text: return []
        n = len(text)

        # DP State: -1 is None
        last_token = [-1] * (n + 1)
        last_token[0] = 0

        # --- LOCAL VARIABLE CACHING (Critical for Loops) ---
        trie_children = self.trie_children
        trie_token_ids = self.trie_token_ids
        is_valid_merge = self._is_valid_merge

        for i in range(n):
            if last_token[i] == -1:
                continue

            prev_token = last_token[i]

            # Start at Root (State 0)
            state = 0

            # Inner Loop: Scan forward
            # We iterate manually to avoid slice creation overhead
            for j in range(i, n):
                byte = text[j]

                # Flattened Trie Access: List[Dict]
                # accessing list by index is faster than object.attr
                children = trie_children[state]

                if byte not in children:
                    break

                state = children[byte]

                # Check if current state ends a token
                # accessing list by index is faster than node.token_id
                tid = trie_token_ids[state]

                if tid != -1:
                    next_pos = j + 1

                    if i == 0:
                        last_token[next_pos] = tid
                    else:
                        if is_valid_merge(prev_token, tid):
                            last_token[next_pos] = tid

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

    # --- STRING OPTIMIZATION ---
    # We cache the STR -> IDs mapping directly.
    # This removes the need to call .encode('utf-8') repeatedly for the same word.
    @lru_cache(maxsize=2 ** 16)
    def encode_chunk_str(self, text: str) -> List[int]:
        return self.encode(text.encode("utf-8"))
