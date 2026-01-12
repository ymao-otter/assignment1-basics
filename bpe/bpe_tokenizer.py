import os
from typing import Iterable, Iterator
import json

from bpe.bpe_training import pre_tokenize


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merge_to_rank = {}
        self.token_to_id = {}
        for i, merge in enumerate(merges):
            self.merge_to_rank[merge] = i
        for id, token in vocab.items():
            self.token_to_id[token] = id
        if self.special_tokens:
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.token_to_id:
                    new_id = len(self.vocab)
                    self.token_to_id[token_bytes] = new_id
                    self.vocab[new_id] = token_bytes

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        import base64
        
        def decode_token(token_str: str) -> bytes:
            """Decode a token string to bytes, handling special encodings."""
            # Handle hex-encoded bytes (128-255): <0xHH>
            if token_str.startswith("<0x") and token_str.endswith(">"):
                hex_str = token_str[3:-1]
                return bytes([int(hex_str, 16)])
            # Handle base64-encoded tokens: <b64:...>
            elif token_str.startswith("<b64:") and token_str.endswith(">"):
                b64_str = token_str[5:-1]
                return base64.b64decode(b64_str)
            # Regular UTF-8 string (includes ASCII bytes 0-127)
            else:
                return token_str.encode("utf-8")
        
        with open(vocab_filepath) as f:
            vocab_dict = json.load(f)
            # Convert string keys to bytes values
            vocab = {}
            for key, value in vocab_dict.items():
                token_bytes = decode_token(key)
                vocab[value] = token_bytes
        
        with open(merges_filepath) as f:
            merges = []
            for line in f:
                line = line.rstrip('\n\r')  # Remove newline characters
                if not line:  # Skip empty lines
                    continue
                # Split by the first space only (maxsplit=1)
                # This handles cases where token1 or token2 might contain spaces
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    token1, token2 = parts
                    merges.append((decode_token(token1), decode_token(token2)))
                # If only 1 part, skip (malformed line)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pre_tokens = pre_tokenize(text, self.special_tokens)
        token_ids = []
        for pre_token_index, pre_token in enumerate(pre_tokens):
            # Check if this is a special token
            if self.special_tokens and pre_token in self.special_tokens:
                # Special tokens are handled directly
                pre_token_bytes = pre_token.encode("utf-8", errors="replace")
                token_ids.append(self.token_to_id[pre_token_bytes])
            else:
                # Regular text goes through BPE encoding
                pre_token_bytes = pre_token.encode("utf-8", errors="replace")
                byte_list = [bytes([b]) for b in list(pre_token_bytes)]

                while True:
                    min_rank_tuple = None
                    for i in range(len(byte_list) - 1):
                        pair = (byte_list[i], byte_list[i + 1])
                        if pair in self.merge_to_rank:
                            rank = self.merge_to_rank[pair]
                            if min_rank_tuple is None or rank < min_rank_tuple[0]:
                                min_rank_tuple = (rank, pair, i)
                    if min_rank_tuple is None:
                        break

                    min_rank, min_rank_pair, min_rank_index = min_rank_tuple
                    merged_bytes = min_rank_pair[0] + min_rank_pair[1]
                    byte_list = (
                        byte_list[:min_rank_index]
                        + [merged_bytes]
                        + byte_list[min_rank_index + 2 :]
                    )

                token_ids.extend([self.token_to_id[token] for token in byte_list])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_list = []
        for id in ids:
            bytes_list.append(self.vocab[id])
        return b"".join(bytes_list).decode("utf-8", errors="replace")
