from collections import Counter, defaultdict
import os
import heapq
from pathlib import Path
from typing import BinaryIO, Generator
import regex
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
END_OF_TEXT = b"<|endoftext|>"


class ReverseTuple:
    """Wrapper for tuple that reverses comparison order for use in min-heap."""

    def __init__(self, t):
        self.t = t

    def __lt__(self, other):
        return self.t > other.t

    def __le__(self, other):
        return self.t >= other.t

    def __gt__(self, other):
        return self.t < other.t

    def __ge__(self, other):
        return self.t <= other.t

    def __eq__(self, other):
        return self.t == other.t

    def __repr__(self):
        return f"ReverseTuple({self.t})"


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


def pre_tokenize(chunk: str, special_tokens: list[str]) -> list[str]:
    pattern = "|".join([regex.escape(t) for t in special_tokens])
    pre_tokens = []
    for sub_chunk in regex.splititer(pattern, chunk):
        for match in regex.finditer(PAT, sub_chunk):
            pre_tokens.append(match.group())
    return pre_tokens


def pre_tokenize_file(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> dict[bytes, int]:
    num_processes = mp.cpu_count()
    vocab = Counter[bytes]()
    with open_binary(input_path) as f:
        boundaries = find_chunk_boundaries(f, num_processes, END_OF_TEXT)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                futures.append(executor.submit(pre_tokenize, chunk, special_tokens))
            # Process futures in order to ensure deterministic behavior
            for future in futures:
                pre_tokens: list[str] = future.result()
                for pre_token in pre_tokens:
                    vocab[pre_token.encode("utf-8", errors="ignore")] += 1
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
    vocab = {0: END_OF_TEXT}
    for token in special_tokens:
        encoded_token = token.encode("utf-8")
        if encoded_token != END_OF_TEXT:
            vocab[len(vocab)] = encoded_token
    merges: list[tuple[bytes, bytes]] = []
    for c in range(256):
        vocab[len(vocab)] = bytes([c])

    # p1, p2
    # max
    # pair -> pre_token list -> new pairs -> add new pairs, remove old pairs
    pre_token_count_arr: list[tuple[list[bytes], int]] = []
    for pre_token, count in pre_tokenized_vocab.items():
        pre_token_count_arr.append(
            ([pre_token[i : i + 1] for i in range(0, len(pre_token))], count)
        )

    pair_counter = Counter[bytes]()
    pair_count_heap: list[tuple[int, ReverseTuple, bytes]] = []

    pair_to_pre_token_index_set = defaultdict[bytes, set[int]](set)

    # Track ALL possible representations of each pair_bytes for tie-breaking
    pair_to_representations: dict[bytes, set[tuple[bytes, bytes]]] = defaultdict(set)

    for pre_token_index, (pre_token, count) in enumerate(pre_token_count_arr):
        for i in range(0, len(pre_token) - 1):
            pair = pre_token[i : i + 2]
            pair_bytes = pair[0] + pair[1]
            pair_counter[pair_bytes] += count
            pair_to_pre_token_index_set[pair_bytes].add(pre_token_index)
            pair_to_representations[pair_bytes].add((pair[0], pair[1]))

    for pair_bytes, count in sorted(pair_counter.items()):
        # Use the lexicographically greatest representation for tie-breaking
        pair_tuple = max(pair_to_representations[pair_bytes])
        heapq.heappush(pair_count_heap, (-count, ReverseTuple(pair_tuple), pair_bytes))

    while len(vocab) < vocab_size:
        most_common_pair = heapq.heappop(pair_count_heap)
        most_common_pair_count = most_common_pair[0]
        most_common_pair_bytes = most_common_pair[2]
        if most_common_pair_count != -pair_counter[most_common_pair_bytes]:
            continue
        vocab[len(vocab)] = most_common_pair_bytes
        # Use the lexicographically greatest representation for recording the merge
        merges.append(max(pair_to_representations[most_common_pair_bytes]))
        pairs_to_update = set[bytes]()
        for index in list(pair_to_pre_token_index_set[most_common_pair_bytes]):
            pre_token, pre_token_count = pre_token_count_arr[index]
            new_pre_token_list = []
            j = 0
            while j < len(pre_token) - 1:
                new_pair = pre_token[j : j + 2]
                new_pair_bytes = new_pair[0] + new_pair[1]
                if new_pair_bytes == most_common_pair_bytes:
                    j += 2
                    new_pre_token_list.append(new_pair_bytes)
                else:
                    new_pre_token_list.append(pre_token[j])
                    j += 1
            if j < len(pre_token):
                new_pre_token_list.append(pre_token[j])

            for j in range(0, len(pre_token) - 1):
                # invalidate old pairs
                old_pair = pre_token[j : j + 2]
                old_pair_bytes = old_pair[0] + old_pair[1]
                pair_counter[old_pair_bytes] -= pre_token_count
                pair_to_pre_token_index_set[old_pair_bytes].discard(index)
                # Track pairs whose counts changed for heap update
                pairs_to_update.add(old_pair_bytes)

            for j in range(0, len(new_pre_token_list) - 1):
                # add new pairs
                new_pair = new_pre_token_list[j : j + 2]
                new_pair_bytes = new_pair[0] + new_pair[1]
                pair_to_representations[new_pair_bytes].add((new_pair[0], new_pair[1]))
                pair_counter[new_pair_bytes] += pre_token_count
                pair_to_pre_token_index_set[new_pair_bytes].add(index)
                # Track this pair for heap update
                pairs_to_update.add(new_pair_bytes)

            pre_token_count_arr[index] = (new_pre_token_list, pre_token_count)

        # Now push all updated pairs onto the heap with their final counts
        for pair_bytes in pairs_to_update:
            # Use the lexicographically greatest representation for tie-breaking
            pair_tuple = max(pair_to_representations[pair_bytes])
            heapq.heappush(
                pair_count_heap,
                (-pair_counter[pair_bytes], ReverseTuple(pair_tuple), pair_bytes),
            )

    return vocab, merges
