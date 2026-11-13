from src.tokenizer import ByteMap


def test_bytemap_roundtrip() -> None:
    """Verifies lossless round-trip encoding and decoding for all 256 bytes.

    Instantiates a ByteMap, converts a contiguous sequence of all byte values
    (0x00 through 0xFF) into their mapped Unicode string representation, and
    asserts that reversing the mapping restores the exact original byte array.
    """
    mapper = ByteMap()
    all_bytes = bytes(range(256))
    unicode_str = mapper.bytes_to_string(all_bytes)
    restored_bytes = mapper.string_to_bytes(unicode_str)
    assert restored_bytes == all_bytes


if __name__ == "__main__":
    test_bytemap_roundtrip()
