from src.tokenizer import Tokenizer


def build_token_dictionary(
    tokenizer: Tokenizer
) -> tuple[dict[int, str], int]:
    """Builds the token-ID-to-string lookup table and locates the EOS ID.

    Decodes every vocabulary ID using the custom Byte-Level BPE tokenizer and
    retrieves the stopping token ID (`<|endoftext|>` or `<|im_end|>`) from
    the loaded special tokens table.

    Args:
        tokenizer: The initialized custom Tokenizer instance.

    Returns:
        A tuple containing:
            - token_table: A mapping of integer token IDs to their
                decoded string values.
            - eos_token_id: The integer ID representing the End of
                Sequence token.

    Raises:
        ValueError: If neither `<|endoftext|>` nor `<|im_end|>` exists in
            the tokenizer's special tokens.
    """
    eos_token_id: int | None = None
    for content in ("<|endoftext|>", "<|im_end|>"):
        if content in tokenizer.special_tokens.str_to_id:
            eos_token_id = tokenizer.special_tokens.str_to_id[content]
            break
    if eos_token_id is None:
        raise ValueError("Could not locate EOS token in added_tokens.")
    token_table: dict[int, str] = {}
    for token_id in tokenizer.id_to_token:
        token_table[token_id] = tokenizer.decode([token_id])
    return (token_table, eos_token_id)
