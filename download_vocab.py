import os
import urllib.request

if __name__ == '__main__':
    url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    filename = "cl100k_base.tiktoken"
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print(f"{filename} already exists.")

if __name__ == '__main__':
    url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    filename = "o200k_base.tiktoken"

    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print(f"{filename} already exists.")
