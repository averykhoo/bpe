"""
Core Byte Pair Encoding (BPE) implementation using Aho-Corasick.

This is an optimized Python port of the "Novel Algorithm" described in the `bpe` Rust crate.

Note that ths takes about 2-3x longer to start up than the trie-based implementation.
But it caches locally to disk so that should help

Why Aho-Corasick?
-----------------
Standard BPE implementations often use a simple Trie or hash lookups to find matching tokens.
This requires checking every position in the text and scanning forward, which can be slow
if there are many overlapping tokens.

Aho-Corasick builds a Finite Automaton with "failure links". It allows us to process the
input text in a SINGLE pass (linear time O(N)). At every byte, the automaton tells us
instantly *which* tokens end at the current position, allowing us to perform the
Dynamic Programming validity check immediately.

Key Features:
1. Exact compatibility with OpenAI's tiktoken.
2. O(N) encoding complexity using Aho-Corasick + Dynamic Programming.
3. Caching of the automaton to speed up initialization.
"""

import base64
import hashlib
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Using __slots__ is critical here. A vocabulary like cl100k has 100,000 tokens.
# This creates hundreds of thousands of nodes. Without __slots__, the memory overhead
# of Python's internal __dict__ per object would consume GBs of RAM.
@dataclass(slots=True)
class ACNode:
    """
    A node in the Aho-Corasick automaton.
    Acts as both a Trie node and a State in the Finite Automaton.
    """
    # Standard Trie children: Byte -> Next Node
    children: Dict[int, "ACNode"] = field(default_factory=dict)
    
    # The token ID if a token ends exactly at this node (Trie logic)
    token_id: Optional[int] = None
    
    # Failure Link: Where to jump if the next character doesn't match a child.
    # This points to the longest proper suffix of the current string that exists in the Trie.
    fail: Optional["ACNode"] = None
    
    # Output List: A list of (token_id, token_len) for ALL tokens that end at this node.
    # This includes the token at this node (if any) AND any tokens reachable via failure links
    # (suffixes). E.g., if we matched "she", this list might contain entries for "she", "he", "e".
    outputs: List[Tuple[int, int]] = field(default_factory=list)


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
        
        # 1. Build the Standard Trie
        # We construct the skeleton of the dictionary first.
        self.root = ACNode()
        for i, token in enumerate(tokens):
            node = self.root
            for byte in token:
                if byte not in node.children:
                    node.children[byte] = ACNode()
                node = node.children[byte]
            node.token_id = i
            # Initialize outputs with the token itself
            node.outputs.append((i, len(token)))
            
        # 2. Build Failure Links (BFS)
        # This converts the Trie into an Aho-Corasick automaton.
        self._build_failure_links()
        
        # 3. Build Split Table
        # Pre-calculates merge rules for the validity check.
        self._build_split_table()

    @classmethod
    def from_tiktoken_file(cls, path: str, use_cache: bool = True) -> "BytePairEncoding":
        """
        Load from a standard OpenAI tiktoken file (base64 encoded list).
        
        Args:
            path: Path to the .tiktoken file.
            use_cache: If True, creates a .cache pickle file. Building the AC automaton
                       in Python is CPU intensive (~2-5s), so caching is highly recommended.
        """
        if not use_cache:
            return cls._load_raw(path)

        # Generate a hash of the file content to ensure the cache is valid
        with open(path, "rb") as f:
            content = f.read()
        
        file_hash = hashlib.md5(content).hexdigest()
        cache_path = f"{path}.{file_hash}.ac.cache"

        # Try loading from cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    # We disable the garbage collector during load for a small speedup
                    # on massive object graphs
                    import gc
                    gc.disable()
                    instance = pickle.load(f)
                    gc.enable()
                    return instance
            except (Exception, pickle.UnpicklingError):
                pass # Fallback to raw load if corrupted

        # Load raw and save cache
        instance = cls._load_raw_from_content(content)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(instance, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
            
        return instance

    @classmethod
    def _load_raw(cls, path: str) -> "BytePairEncoding":
        with open(path, "rb") as f:
            return cls._load_raw_from_content(f.read())

    @classmethod
    def _load_raw_from_content(cls, content: bytes) -> "BytePairEncoding":
        tokens = []
        for line in content.splitlines():
            if not line.strip(): continue
            parts = line.split()
            tokens.append(base64.b64decode(parts[0]))
        return cls(tokens)

    def _build_failure_links(self):
        """
        Constructs failure links using Breadth-First Search (BFS).
        
        Standard AC Algorithm:
        1. Root's children fail to Root.
        2. For other nodes, `child.fail` = `parent.fail.children[char]`.
        3. `child.outputs` extends `child.fail.outputs`.
        """
        queue = []
        
        # Initialize depth 1 (immediate children of root)
        for node in self.root.children.values():
            node.fail = self.root
            queue.append(node)
            
        self.root.fail = self.root 

        while queue:
            current = queue.pop(0)
            
            for byte, child in current.children.items():
                # Trace back failure links until we find a node that has a transition for 'byte'
                f = current.fail
                while f is not self.root and byte not in f.children:
                    f = f.fail
                
                if byte in f.children:
                    child.fail = f.children[byte]
                else:
                    child.fail = self.root
                
                # Output Propagation:
                # If the node we failed to is a valid token (or represents the end of one),
                # then the current node also represents the end of that token (as a suffix).
                if child.fail.outputs:
                    child.outputs.extend(child.fail.outputs)
                
                queue.append(child)

    def _find_longest_prefix(self, bytes_seq: bytes) -> Optional[int]:
        """
        Finds the token_id of the longest prefix of bytes_seq.
        Used only during initialization for split table construction.
        """
        # Note: We stick to standard Trie traversal here as it's simple and
        # only runs once.
        node = self.root
        last_valid_token = None
        for byte in bytes_seq:
            if byte not in node.children: break
            node = node.children[byte]
            if node.token_id is not None:
                last_valid_token = node.token_id
        return last_valid_token

    def _build_split_table(self):
        """
        Reverse-engineers the merge tree (which tokens combine to form larger tokens).
        Essential for the validity check algorithm.
        """
        self.split_table = [(i, i) for i in range(len(self.all_tokens))]
        self.pair_lookup = {}

        for i, token in enumerate(self.all_tokens):
            if len(token) == 1: continue
            
            curr_prefix_id = self._find_longest_prefix(token[:-1])
            
            while curr_prefix_id is not None:
                prefix_len = len(self.all_tokens[curr_prefix_id])
                suffix = token[prefix_len:]
                
                if suffix in self.token_map:
                    suffix_id = self.token_map[suffix]
                    if curr_prefix_id < i and suffix_id < i:
                        if self._is_valid_merge(curr_prefix_id, suffix_id):
                            self.pair_lookup[(curr_prefix_id, suffix_id)] = i
                            self.split_table[i] = (curr_prefix_id, suffix_id)
                            break
                
                curr_prefix_id = self._find_longest_prefix(token[:prefix_len-1])

    def _is_valid_merge(self, t1: int, t2: int) -> bool:
        """
        Verifies if merging t1 and t2 is consistent with the BPE priority rules.
        This check prevents the "greedy trap" where a locally optimal merge
        might block a globally correct merge.
        """
        limit = float('inf')
        curr_t1, curr_t2 = t1, t2
        
        while True:
            if (curr_t1, curr_t2) in self.pair_lookup:
                if self.pair_lookup[(curr_t1, curr_t2)] < limit:
                    return False
            
            if curr_t1 > curr_t2:
                limit = curr_t1
                t1_left, t1_right = self.split_table[curr_t1]
                curr_t1 = t1_right
                
                if curr_t1 == limit: 
                    limit = curr_t2 + 1
                    t2_left, t2_right = self.split_table[curr_t2]
                    curr_t2 = t2_left 
                    if curr_t2 + 1 == limit: 
                        return True
                continue
            
            limit = curr_t2 + 1
            t2_left, t2_right = self.split_table[curr_t2]
            if t2_left == curr_t2: 
                return True
            
            curr_t2 = t2_left
            if curr_t2 + 1 == limit:
                limit = curr_t1
                t1_left, t1_right = self.split_table[curr_t1]
                curr_t1 = t1_right
                if curr_t1 == limit: 
                    return True

    def decode_tokens(self, tokens: List[int]) -> bytes:
        return b"".join(self.all_tokens[t] for t in tokens)

    def encode(self, text: bytes) -> List[int]:
        """
        Encodes text using the 'Novel Algorithm' backed by Aho-Corasick.
        
        Algorithm:
        1. Scan the text in a single pass (O(N)).
        2. At every byte `i`, the Automaton provides all dictionary words 
           that *end* at `i`.
        3. Use Dynamic Programming (via `last_token` dict) to determine if 
           a word ending at `i` connects validly to a previous token.
        4. Reconstruct the path backwards.
        """
        if not text: return []
        n = len(text)
        
        # last_token[i] stores the token_id that ends at position i.
        # '0: None' indicates the start of the string is a valid boundary.
        last_token: Dict[int, int] = {0: None}
        
        # State of the automaton
        node = self.root
        
        for i, byte in enumerate(text):
            # 1. Follow failure links if the current byte doesn't match a child.
            # This "rewinds" the state to the longest matching suffix.
            while node is not self.root and byte not in node.children:
                node = node.fail
            
            # 2. Advance state if possible
            if byte in node.children:
                node = node.children[byte]
            
            # 3. Process matches ending at this position (i + 1)
            # node.outputs contains tuples (token_id, length)
            for token_id, token_len in node.outputs:
                # The start of this token in the text would be:
                start_pos = (i + 1) - token_len
                
                # Dynamic Programming Check:
                # Do we have a valid token ending exactly where this one starts?
                if start_pos not in last_token:
                    continue
                
                # BPE Validity Check:
                # Is this merge allowed given the greedy history?
                is_valid = False
                if start_pos == 0:
                    is_valid = True
                else:
                    prev_token = last_token[start_pos]
                    if self._is_valid_merge(prev_token, token_id):
                        is_valid = True
                
                # If valid, record it. 
                # Corollary IIa: There is only one valid segmentation. We overwrite.
                if is_valid:
                    last_token[i + 1] = token_id

        # 4. Backward Pass: Reconstruct
        if n not in last_token:
            return []

        encoded = []
        curr = n
        while curr > 0:
            t = last_token[curr]
            encoded.append(t)
            curr -= len(self.all_tokens[t])
            
        return encoded[::-1]