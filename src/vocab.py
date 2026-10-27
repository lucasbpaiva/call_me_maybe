import json
from llm_sdk import Small_LLM_Model


def build_token_dictionary(
    model: Small_LLM_Model
) -> tuple[dict[int, str], int]:
    """Loads the vocabulary, decodes tokens, and extracts the EOS token ID.

    Reads the raw tokenizer vocabulary JSON file located by the SDK, decodes
    each token ID into its true string representation using Byte-Level BPE, and
    locates the required stopping token ID (`<|endoftext|>` or `<|im_end|>`).

    Args:
        model (Small_LLM_Model): The initialized SDK model instance used to
            fetch the vocabulary path and decode individual tokens.

    Returns:
        Tuple[Dict[int, str], int]: A tuple containing:
            - token_table (Dict[int, str]): A mapping of integer token IDs to
              their decoded string values.
            - eos_token_id (int): The integer ID representing the End of
              Sequence token.

    Raises:
        RuntimeError: If the vocabulary JSON file cannot be opened or parsed.
        ValueError: If neither `<|endoftext|>` nor `<|im_end|>` exists in
            the tokenizer vocabulary.
    """
    vocab_path = model.get_path_to_vocab_file()
    tokenizer_path = model.get_path_to_tokenizer_file()
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer files: {e}")
    eos_token_id = None
    for token_data in tokenizer_data.get("added_tokens", []):
        if token_data.get("content") in ("<|endoftext|>", "<|im_end|>"):
            eos_token_id = token_data.get("id")
            break
    if eos_token_id is None:
        raise ValueError(
            "Could not locate EOS token in added_tokens."
        )
    token_table: dict[int, str] = {}
    for token_str, token_id in raw_vocab.items():
        token_table[token_id] = model.decode([token_id])
    return (token_table, eos_token_id)
