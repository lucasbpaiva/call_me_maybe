import json
from typing import Any
from pathlib import Path
from pydantic import BaseModel, Field


def build_byte_to_unicode() -> dict[int, str]:
    """Builds a bijective mapping from all 256 byte values to Unicode chars.

    Maps printable ASCII (33-126) and extended Latin-1 characters (161-172,
    174-255) directly to their corresponding Unicode code points. Non-printable
    control bytes and whitespace are shifted to higher Unicode code points
    starting at 256 so every byte has a visible, safe text representation.

    Returns:
        A dictionary mapping integer byte values (0-255) to single-character
        Unicode strings.
    """
    printable_chars = list(range(33, 127))
    extended_ascii = list(range(161, 173)) + list(range(174, 256))
    valid_bytes = printable_chars + extended_ascii
    byte_to_unicode = {}
    for b in valid_bytes:
        byte_to_unicode[b] = chr(b)
    n = 0
    for b in range(256):
        if b not in byte_to_unicode:
            byte_to_unicode[b] = chr(256 + n)
            n += 1
    return byte_to_unicode


class ByteMap(BaseModel):
    """Maps raw bytes to Unicode chars and vice-versa for BPE tokenization.

    Provides bidirectional conversion between raw byte sequences and the
    printable Unicode character representation used internally by byte-level
    BPE vocabularies and merge tables.

    Attributes:
        byte_to_char: Forward lookup mapping integer byte values (0-255) to
            printable Unicode characters.
        char_to_byte: Reverse lookup mapping printable Unicode characters back
            to their original integer byte values.
    """
    byte_to_char: dict[int, str] = Field(default_factory=build_byte_to_unicode)
    char_to_byte: dict[str, int] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Populates the reverse character-to-byte lookup after initialization.

        Args:
            __context: Optional validation context passed by Pydantic.
        """
        self.char_to_byte = {c: byte for byte, c in self.byte_to_char.items()}

    def bytes_to_str(self, b_data: bytes) -> str:
        """Converts raw bytes to the BPE internal string representation.

        Args:
            b_data: A sequence of raw UTF-8 encoded bytes.

        Returns:
            A string composed exclusively of mapped printable Unicode symbols.
        """
        return "".join(self.byte_to_char[b] for b in b_data)

    def str_to_bytes(self, text: str) -> bytes:
        """Converts BPE internal string representation back to raw bytes.

        Args:
            text: A string of mapped BPE Unicode symbols.

        Returns:
            The reconstructed sequence of raw bytes.
        """
        return bytes(self.char_to_byte[c] for c in text)


class AddedToken(BaseModel):
    """Represents a special or added token defined outside the BPE vocab."""
    id: int
    content: str
    special: bool


class BPEModelConfig(BaseModel):
    """Validates the core BPE structures from tokenizer.json."""
    vocab: dict[str, int]
    merges: list[list[str]]


class TokenizerData(BaseModel):
    """Validates the root structure of tokenizer.json."""
    model: BPEModelConfig
    added_tokens: list[AddedToken]


class Tokenizer(BaseModel):
    """Main Tokenizer class handling BPE decode operations."""
    byte_map: ByteMap = Field(default_factory=ByteMap)
    token_to_id: dict[str, int] = Field(default_factory=dict)
    id_to_token: dict[int, str] = Field(default_factory=dict)
    special_tokens: dict[int, AddedToken] = Field(default_factory=dict)

    @classmethod
    def load_from_file(cls, path: str | Path) -> "Tokenizer":
        """Loads, validates, and initializes the tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.loads(f.read())
        parsed = TokenizerData.model_validate(raw_data)
        id_to_token = {v: k for k, v in parsed.model.vocab.items()}
        special_tokens = {t.id: t for t in parsed.added_tokens}
        return cls(
            token_to_id=parsed.model.vocab,
            id_to_token=id_to_token,
            special_tokens=special_tokens
        )

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of token IDs back into a standard UTF-8 string."""
        decoded_text = ""
        current_symbols: list[str] = []

        def flush_symbols() -> None:
            nonlocal decoded_text, current_symbols
            if current_symbols:
                b_data = self.byte_map.str_to_bytes("".join(current_symbols))
                decoded_text += b_data.decode("utf-8", errors="replace")
                current_symbols = []

        for token_id in ids:
            if token_id in self.special_tokens:
                token_data = self.special_tokens[token_id]
                if skip_special_tokens and token_data.special:
                    continue
                flush_symbols()
                decoded_text += token_data.content
            else:
                symbol_str = self.id_to_token.get(token_id)
                if symbol_str is None:
                    raise ValueError(f"Unknown token ID: {token_id}")
                current_symbols.append(symbol_str)
        flush_symbols()
        return decoded_text
