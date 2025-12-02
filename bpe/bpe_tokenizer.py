import os
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        raise NotImplementedError

    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError
