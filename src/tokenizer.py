from typing import Any
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

    def bytes_to_string(self, b_data: bytes) -> str:
        """Converts raw bytes to the BPE internal string representation.

        Args:
            b_data: A sequence of raw UTF-8 encoded bytes.

        Returns:
            A string composed exclusively of mapped printable Unicode symbols.
        """
        return "".join(self.byte_to_char[b] for b in b_data)

    def string_to_bytes(self, text: str) -> bytes:
        """Converts BPE internal string representation back to raw bytes.

        Args:
            text: A string of mapped BPE Unicode symbols.

        Returns:
            The reconstructed sequence of raw bytes.
        """
        return bytes(self.char_to_byte[c] for c in text)
