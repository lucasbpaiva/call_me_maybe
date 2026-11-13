from src.tokenizer import ByteMap


def test_bytemap_roundtrip() -> None:
    """Verifies lossless round-trip encoding and decoding for all 256 bytes.

    Instantiates a ByteMap, converts a contiguous sequence of all byte values
    (0x00 through 0xFF) into their mapped Unicode string representation, and
    asserts that reversing the mapping restores the exact original byte array.
    """
    mapper = ByteMap()
    assert mapper.byte_to_char[32] == "Ġ"
    assert mapper.byte_to_char[10] == "Ċ"
    assert len(set(mapper.byte_to_char.values())) == 256
    all_bytes = bytes(range(256))
    unicode_str = mapper.bytes_to_str(all_bytes)
    restored_bytes = mapper.str_to_bytes(unicode_str)
    assert restored_bytes == all_bytes
