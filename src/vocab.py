import json
from llm_sdk.llm_sdk import Small_LLM_Model


def build_token_dictionary(model: Small_LLM_Model) -> dict[int, str]:
    """
    Build the token dictionary.

    Loads the model's vocabulary and decodes each token ID to its true string
    representation.
    """
    vocab_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load vocab file: {e}")
    token_table: dict[int, str] = {}
    for token_str, token_id in raw_vocab.items():
        clean_text = model.decode([token_id])
        token_table[token_id] = clean_text
    return token_table
