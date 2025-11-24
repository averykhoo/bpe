import time

from tokenizer import get_cl100k_base
from tokenizer import get_o200k_base

# 1. Initialize
# This loads the file and computes the split table.
# This takes ~15 seconds on a modern CPU due to pure Python overhead.

if __name__ == '__main__':
    print("Loading cl100k tokenizer (this takes roughly 2 seconds)...")
    start_time = time.time()
    enc1 = get_cl100k_base("cl100k_base.tiktoken")

    print(f"Loaded in {time.time() - start_time:.2f} seconds.")

    # 2. Encode Text
    text = "Hello, world! This is a test of the 100% Python BPE port." * 100000
    # print(f"\nInput: '{text}'")

    ids = enc1.encode(text)
    # print(f"Token IDs: {ids}")
    print(f"Count: {len(ids)}")

    # 3. Decode Tokens
    # decoded_text = enc1.decode(ids)
    # print(f"Decoded: '{decoded_text}'")

    # 4. Prove correctness (Compare specific edge cases)
    # The word " world" (with space) usually maps to a single token in cl100k
    print("\n--- Edge Case Checks ---")
    print(f"' world': {enc1.encode(' world')}")
    print(f"'world':  {enc1.encode('world')}")  # Different token without space

    # 5. Unicode Test
    unicode_text = "I ❤️ 🐍"
    print(f"\nUnicode: {unicode_text}")
    print(f"IDs: {enc1.encode(unicode_text)}")
    print(f"Finished in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    print("Loading o200k tokenizer (this takes roughly 5 seconds)...")
    start_time = time.time()
    enc2 = get_o200k_base("o200k_base.tiktoken")

    print(f"Loaded in {time.time() - start_time:.2f} seconds.")

    # 2. Encode Text
    text = "Hello, world! This is a test of the 100% Python BPE port." * 100000
    # print(f"\nInput: '{text}'")

    ids = enc2.encode(text)
    # print(f"Token IDs: {ids}")
    print(f"Count: {len(ids)}")

    # 3. Decode Tokens
    # decoded_text = enc2.decode(ids)
    # print(f"Decoded: '{decoded_text}'")

    # 4. Prove correctness (Compare specific edge cases)
    # The word " world" (with space) usually maps to a single token in cl100k
    print("\n--- Edge Case Checks ---")
    print(f"' world': {enc2.encode(' world')}")
    print(f"'world':  {enc2.encode('world')}")  # Different token without space

    # 5. Unicode Test
    unicode_text = "I ❤️ 🐍"
    print(f"\nUnicode: {unicode_text}")
    print(f"IDs: {enc2.encode(unicode_text)}")
    print(f"Finished in {time.time() - start_time:.2f} seconds.")
