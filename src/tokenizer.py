import json
from typing import Any
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError


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
    """Represents a special or added token defined outside the BPE vocabulary.

    Attributes:
        id: The unique integer token ID assigned to this added token.
        content: The literal text string represented by the token.
        special: Indicates whether this token is treated as a control or
            special token that can be omitted during decoding.
    """
    id: int
    content: str
    special: bool


class BPEModelConfig(BaseModel):
    """Validates the core Byte Pair Encoding structures from tokenizer.json.

    Attributes:
        vocab: A dictionary mapping BPE internal symbol strings to integer
            token IDs.
        merges: An ordered list of symbol pairs defining the BPE merge priority
            rules.
    """
    vocab: dict[str, int]
    merges: list[list[str]]


class TokenizerData(BaseModel):
    """Validates the root structure of a Hugging Face tokenizer.json file.

    Attributes:
        model: The core BPE model config containing vocabulary and merges.
        added_tokens: A list of special or supplementary tokens defined outside
            the base vocabulary.
    """
    model: BPEModelConfig
    added_tokens: list[AddedToken]


class BPE(BaseModel):
    """Executes deterministic Byte Pair Encoding merges on mapped words.

    Attributes:
        merges: Ordered list of symbol pairs defining merge priority.
        ranks: Lookup mapping adjacent symbol pairs to their priority rank.
        cache: Memoization table mapping mapped input words to merged symbols.
    """

    merges: list[list[str]] = Field(default_factory=list)
    ranks: dict[tuple[str, str], int] = Field(default_factory=dict)
    cache: dict[str, list[str]] = Field(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:
        """Builds a pair-to-rank lookup table from the ordered merges list.

        Args:
            __context: Optional validation context passed by Pydantic.
        """
        self.ranks = {
            (pair[0], pair[1]): rank
            for rank, pair in enumerate(self.merges)
        }

    def merge_word(self, word: str) -> list[str]:
        """Repeatedly merges the lowest-rank adjacent pair in a mapped word.

        Args:
            word: A single pre-tokenized word already converted via ByteMap.

        Returns:
            A list of merged BPE vocabulary symbol strings.
        """
        if not word:
            return []
        if word in self.cache:
            return self.cache[word]
        symbols = list(word)
        while len(symbols) > 1:
            best_pair: tuple[str, str] | None = None
            best_rank: int | None = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.ranks.get(pair)
                if rank is not None and (
                    best_rank is None or rank < best_rank
                ):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):
                    merged.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        self.cache[word] = symbols
        return symbols


class Tokenizer(BaseModel):
    """Main Tokenizer class handling BPE vocabulary lookups and decoding.

    Attributes:
        byte_map: Bidirectional lookup table converting between raw bytes and
            BPE printable Unicode symbols.
        token_to_id: Forward lookup mapping BPE internal symbol strings to
            integer token IDs.
        id_to_token: Reverse lookup mapping integer token IDs to BPE internal
            symbol strings.
        special_tokens: A dictionary mapping special token IDs to their
            corresponding AddedToken metadata.
    """
    byte_map: ByteMap = Field(default_factory=ByteMap)
    bpe: BPE = Field(default_factory=BPE)
    token_to_id: dict[str, int] = Field(default_factory=dict)
    id_to_token: dict[int, str] = Field(default_factory=dict)
    special_tokens: dict[int, AddedToken] = Field(default_factory=dict)

    @classmethod
    def load_from_file(cls, path: str | Path) -> "Tokenizer":
        """Loads, validates, and initializes the tokenizer from a JSON file.

        Args:
            path: File path to the tokenizer.json configuration file.

        Returns:
            An initialized Tokenizer instance populated with vocabulary and
            special token tables.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            parsed = TokenizerData.model_validate(raw_data)
        except (OSError, json.JSONDecodeError, ValidationError) as err:
            msg = f"Failed to load or validate tokenizer from '{path}': {err}"
            raise RuntimeError(msg) from err
        byte_map = ByteMap()
        token_to_id = parsed.model.vocab
        id_to_token = {v: k for k, v in token_to_id.items()}
        special_tokens = {t.id: t for t in parsed.added_tokens}
        bpe = BPE(merges=parsed.model.merges)
        if len(id_to_token) != len(token_to_id):
            raise ValueError("Duplicate token IDs detected in vocabulary.")
        if set(id_to_token).intersection(special_tokens):
            msg = "Overlap detected between vocab IDs and added_tokens."
            raise ValueError(msg)
        valid_alphabet = set(byte_map.char_to_byte)
        for token_str in token_to_id:
            if not set(token_str).issubset(valid_alphabet):
                raise ValueError(
                    f"Vocab token {token_str!r} contains unmapped characters."
                )
        return cls(
            byte_map=byte_map,
            bpe=bpe,
            token_to_id=parsed.model.vocab,
            id_to_token=id_to_token,
            special_tokens=special_tokens
        )

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decodes a sequence of token IDs back into a standard UTF-8 string.

        Accumulates consecutive ordinary BPE symbols and converts them into a
        contiguous byte stream before decoding UTF-8, preserving multi-byte
        characters that span across multiple tokens. Added tokens bypass byte
        mapping and are injected directly as literal text.

        Args:
            ids: A list of integer token IDs to decode.
            skip_special_tokens: If True, omits tokens marked as special from
                the final output string. Defaults to True.

        Returns:
            The reconstructed UTF-8 text string.

        Raises:
            ValueError: If an encountered token ID does not exist in either the
                base vocabulary or special tokens table.
        """
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
