from collections import Counter, defaultdict
import os
import heapq
import json
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


def pre_tokenize(chunk: str, special_tokens: list[str] | None = None) -> list[str]:
    pre_tokens = []
    if special_tokens:
        # Sort special tokens topologically: if A contains B as substring, A comes before B
        # This ensures longer/containing tokens are matched first
        def sort_key(token):
            # Primary: number of other tokens this token contains (descending)
            # Secondary: length (descending)
            contains_count = sum(
                1 for other in special_tokens if other != token and other in token
            )
            return (-contains_count, -len(token))

        sorted_special_tokens = sorted(special_tokens, key=sort_key)
        # Use capturing group to preserve special tokens in split
        pattern = "(" + "|".join([regex.escape(t) for t in sorted_special_tokens]) + ")"
        for part in regex.splititer(pattern, chunk):
            if not part:  # Skip empty strings
                continue
            if part in special_tokens:
                # Special token - add as is
                pre_tokens.append(part)
            else:
                # Regular text - apply PAT regex
                for match in regex.finditer(PAT, part):
                    pre_tokens.append(match.group())
    else:
        for match in regex.finditer(PAT, chunk):
            pre_tokens.append(match.group())
    return pre_tokens


def _process_chunk_from_file(
    input_path: str | os.PathLike, start: int, end: int, special_tokens: list[str]
) -> Counter[bytes]:
    """Worker function that reads and processes a chunk directly from file."""
    vocab = Counter[bytes]()
    with open_binary(input_path) as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        pre_tokens = pre_tokenize(chunk, special_tokens)
        for pre_token in pre_tokens:
            # Filter out special tokens - they should not be part of the training vocab
            if pre_token not in special_tokens:
                vocab[pre_token.encode("utf-8", errors="ignore")] += 1
    return vocab


def pre_tokenize_file(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> dict[bytes, int]:
    num_processes = 2  # mp.cpu_count()
    vocab = Counter[bytes]()
    with open_binary(input_path) as f:
        boundaries = find_chunk_boundaries(f, num_processes, END_OF_TEXT)

    # Process chunks in parallel without loading all into main process memory
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            futures.append(
                executor.submit(
                    _process_chunk_from_file, input_path, start, end, special_tokens
                )
            )
        # Merge results from all workers
        for future in futures:
            chunk_vocab = future.result()
            vocab.update(chunk_vocab)
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

    # Track pairs by their actual token composition, not concatenated bytes
    pair_counter = Counter[tuple[bytes, bytes]]()
    pair_count_heap: list[tuple[int, ReverseTuple, tuple[bytes, bytes]]] = []

    # Map each pair to the set of pre-token indices where it appears
    pair_to_pre_token_index_set = defaultdict[tuple[bytes, bytes], set[int]](set)

    for pre_token_index, (pre_token, count) in enumerate(pre_token_count_arr):
        for i in range(0, len(pre_token) - 1):
            pair_tuple = (pre_token[i], pre_token[i + 1])
            pair_counter[pair_tuple] += count
            pair_to_pre_token_index_set[pair_tuple].add(pre_token_index)

    for pair_tuple, count in sorted(pair_counter.items()):
        heapq.heappush(pair_count_heap, (-count, ReverseTuple(pair_tuple), pair_tuple))

    while len(vocab) < vocab_size:
        most_common_pair = heapq.heappop(pair_count_heap)
        most_common_pair_count = most_common_pair[0]
        most_common_pair_tuple = most_common_pair[2]
        if most_common_pair_count != -pair_counter[most_common_pair_tuple]:
            continue
        # The merged bytes is the concatenation of the two tokens
        merged_bytes = most_common_pair_tuple[0] + most_common_pair_tuple[1]
        vocab[len(vocab)] = merged_bytes
        merges.append(most_common_pair_tuple)
        pairs_to_update = set[tuple[bytes, bytes]]()
        for index in list(pair_to_pre_token_index_set[most_common_pair_tuple]):
            pre_token, pre_token_count = pre_token_count_arr[index]
            new_pre_token_list = []
            j = 0
            while j < len(pre_token) - 1:
                # Check if this pair matches the one we're merging
                if (pre_token[j], pre_token[j + 1]) == most_common_pair_tuple:
                    j += 2
                    new_pre_token_list.append(merged_bytes)
                else:
                    new_pre_token_list.append(pre_token[j])
                    j += 1
            if j < len(pre_token):
                new_pre_token_list.append(pre_token[j])

            for j in range(0, len(pre_token) - 1):
                # invalidate old pairs
                old_pair_tuple = (pre_token[j], pre_token[j + 1])
                pair_counter[old_pair_tuple] -= pre_token_count
                pair_to_pre_token_index_set[old_pair_tuple].discard(index)
                # Track pairs whose counts changed for heap update
                pairs_to_update.add(old_pair_tuple)

            for j in range(0, len(new_pre_token_list) - 1):
                # add new pairs
                new_pair_tuple = (new_pre_token_list[j], new_pre_token_list[j + 1])
                pair_counter[new_pair_tuple] += pre_token_count
                pair_to_pre_token_index_set[new_pair_tuple].add(index)
                # Track this pair for heap update
                pairs_to_update.add(new_pair_tuple)

            pre_token_count_arr[index] = (new_pre_token_list, pre_token_count)

        # Now push all updated pairs onto the heap with their final counts
        for pair_tuple in pairs_to_update:
            heapq.heappush(
                pair_count_heap,
                (-pair_counter[pair_tuple], ReverseTuple(pair_tuple), pair_tuple),
            )

    return vocab, merges


def train_bpe_to_files(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    vocab_output_path: str | os.PathLike,
    merges_output_path: str | os.PathLike,
    **kwargs,
) -> None:
    """Train a BPE tokenizer and save the vocabulary and merges to files.

    This function calls run_train_bpe and persists the results in standard BPE format:
    - Vocabulary saved as JSON (token string -> token ID mapping)
    - Merges saved as text file (space-separated token pairs, one per line)

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
        vocab_output_path (str | os.PathLike): Path where the vocabulary JSON file will be saved.
        merges_output_path (str | os.PathLike): Path where the merges text file will be saved.
        **kwargs: Additional arguments passed to run_train_bpe.

    Returns:
        None. Results are written to the specified output files.
    """
    # Train the BPE tokenizer
    vocab, merges = run_train_bpe(input_path, vocab_size, special_tokens, **kwargs)

    # Save vocabulary as JSON (inverted: token string -> token ID)
    # Fix: Only encode bytes 128-255 with hex, let UTF-8 handle 0-127
    import base64
    vocab_dict = {}
    for token_id, token_bytes in vocab.items():
        # Handle single bytes (0-255)
        if len(token_bytes) == 1:
            byte_val = token_bytes[0]
            if byte_val >= 128:
                # Bytes 128-255 are not valid UTF-8, use hex encoding
                token_str = f"<0x{byte_val:02X}>"
            else:
                # Bytes 0-127 are valid ASCII/UTF-8, decode normally
                # JSON will automatically escape special chars like \n, ", etc.
                token_str = token_bytes.decode("utf-8")
        else:
            # Multi-byte sequences
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # For non-UTF8 sequences, use base64 with special prefix
                token_str = "<b64:" + base64.b64encode(token_bytes).decode('ascii') + ">"
        
        vocab_dict[token_str] = token_id

    with open(vocab_output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, indent=4, ensure_ascii=True)

    # Save merges as space-separated text file
    with open(merges_output_path, "w", encoding="utf-8") as f:
        def encode_token(token_bytes):
            """Encode token bytes to string for merges file."""
            if len(token_bytes) == 1:
                byte_val = token_bytes[0]
                if byte_val >= 128:
                    return f"<0x{byte_val:02X}>"
                else:
                    return token_bytes.decode("utf-8")
            else:
                try:
                    return token_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    return "<b64:" + base64.b64encode(token_bytes).decode('ascii') + ">"
        
        for token1, token2 in merges:
            token1_str = encode_token(token1)
            token2_str = encode_token(token2)
            f.write(f"{token1_str} {token2_str}\n")
