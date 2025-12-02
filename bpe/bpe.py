from collections import Counter, defaultdict
import os

from pathlib import Path
from typing import BinaryIO, Generator
import regex
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
END_OF_TEXT = b"<|endoftext|>"


def open_binary(input_path: str | os.PathLike) -> BinaryIO:
    path = Path(input_path)  # Normalize str or PathLike
    return path.open("rb")  # Equivalent to open(path, "rb")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


def pre_tokenize(chunk: str, special_tokens: list[str]) -> Generator[str, None, None]:
    pattern = "|".join([regex.escape(t) for t in special_tokens])
    for sub_chunk in regex.splititer(pattern, chunk):
        for match in regex.finditer(PAT, sub_chunk):
            yield match.group()


def pre_tokenize_file(input_path: str | os.PathLike, special_tokens: list[str]) ->dict[str, int]:
    num_processes = mp.cpu_count()
    vocab = Counter()
    with open_binary(input_path) as f:
        boundaries = find_chunk_boundaries(f, num_processes, END_OF_TEXT)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                futures.append(executor.submit(pre_tokenize, chunk, special_tokens))
            for future in as_completed(futures):
                pre_tokens = future.result()
                for pre_token in pre_tokens:
                    vocab[pre_token] += 1
    return vocab

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    pre_tokenized_vocab = pre_tokenize_file(input_path, special_tokens)
    vocab = {END_OF_TEXT: 0}
    for c in range(1, 256):
        vocab[chr(c).encode("utf-8")] = len(vocab)
    for token in special_tokens:
        encoded_token = token.encode("utf-8")
        if encoded_token != END_OF_TEXT:
            vocab[encoded_token] = len(vocab)
    # p1, p2
    # max
    # pair -> pre_token list -> new pairs -> add new pairs, remove old pairs
    pre_token_count_arr = []
    for pre_token, count in pre_tokenized_vocab.items():
        pre_token_count_arr.append((list(pre_token.encode("utf-8")), count))
    
    pair_counter = Counter()
    pair_to_pre_token_index_set = defaultdict(set)

    for i, (pre_token, count) in enumerate(pre_token_count_arr):
        for i in range(0, len(pre_token) - 1):
            pair = pre_token[i:i+2]
            pair_str = ''.join(pair)
            pair_counter[pair_str] += count
            pair_to_pre_token_index_set[pair_str].add(i)

    while len(vocab) < vocab_size:
        most_common_pair = max(pair_counter.items(), key=lambda x: (x[1], x[0]))
        pair = most_common_pair[0]
        dec = 0
        new_pair = pair[0] + pair[1]
        new_pair_list = 
        for index in pair_to_pre_token_index_set[pair]:
            pre_token, pre_token_count = pre_token_count_arr[index]
            new_pre_token_list = []
            for j in range(0, len(pre_token) - 1):

        
    
    return vocab, merges
