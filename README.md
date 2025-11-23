[//]: # (
Here is a `README.md` for the Python implementation we just built. 
It gives full credit to the original Rust architecture while explaining how to use this standalone Python version.
)

# Pure Python OpenAI BPE Tokenizer

A pure Python port of the high-performance Byte Pair Encoding (BPE) algorithm detailed in
[this blog post](https://github.blog/ai-and-ml/llms/so-many-tokens-so-little-time-introducing-a-faster-more-flexible-byte-pair-tokenizer/).
The original source code can be found [here](https://github.com/github/rust-gems/tree/main/crates/bpe) (bpe algorithm) 
and [here](https://github.com/github/rust-gems/tree/main/crates/bpe-openai) (openai tokenizer)

## 🔗 Original Source

This code is based entirely on the logic from the **`bpe`** and **`bpe-openai`** crates found 
[here](https://github.com/github/rust-gems).

While the original Rust implementation is optimized for production use, this Python port is designed for:

* **Education:** Understanding how GPT-4 tokenization works under the hood (Regex split + BPE merge).
* **Portability:** Running in environments where compiling Rust extensions or installing binary wheels is difficult.
* **Zero-Dependency:** Depends only on `regex` (for PCRE patterns) and the standard library.

## 📂 Files

* **`byte_pair_encoding.py`**: The core BPE logic. Implements the Trie structure,
  the valid merge check (Dynamic Programming), and the `.tiktoken` file loader.
* **`tokenizer.py`**: The wrapper that handles OpenAI's specific Regex splitting patterns
  (`cl100k_base`, `o200k_base`) and normalization.
* **`download_vocab.py`**: A helper to fetch the public vocabulary files.

## 🛠️ Installation

You only need the `regex` library
(because Python's standard `re` module does not support the specific Unicode properties used by GPT-4).

```bash
pip install regex
```

## 🚀 Usage

### 1. Download the Vocabulary

First, download the official dictionary file from OpenAI.

```python
# Run this once or use the provided download script
import urllib.request

urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
                           "cl100k_base.tiktoken")
```

### 2. Encode and Decode

```python
from tokenizer import get_cl100k_base

# Initialize the tokenizer
# Note: This takes ~1-2 seconds to load the 100k token vocabulary.
enc = get_cl100k_base("cl100k_base.tiktoken")

text = "Hello, world! This is a pure Python BPE."

# Encode to integers
tokens = enc.encode(text)
print(f"IDs: {tokens}")
# Output: [9906, 11, 1917, 0, 1115, 374, 264, 10748, 13325, 426, 1777, 13]

# Decode back to string
decoded = enc.decode(tokens)
print(f"Decoded: {decoded}")
```

### 3. Using GPT-4o (o200k_base)

If you download the `o200k_base.tiktoken` file, you can use the newer tokenizer:

```python
from tokenizer import get_o200k_base

enc = get_o200k_base("o200k_base.tiktoken")
```
