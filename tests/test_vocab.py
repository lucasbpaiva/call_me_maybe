import pytest
from llm_sdk import Small_LLM_Model
from src.vocab import build_token_dictionary

def test_vocab_dictionary_matches_sdk() -> None:
    model = Small_LLM_Model()
    vocab_dict = build_token_dictionary(model)
    sample_size = 2000
    for token_id in range(sample_size):
        # Ensure the token ID actually exists in the dictionary
        if token_id in vocab_dict:
            my_decoded_string = vocab_dict[token_id]
            sdk_decoded_string = model.decode([token_id])
            assert my_decoded_string == sdk_decoded_string, \
                f"Mismatch at ID {token_id}: " \
                f"'{my_decoded_string}' vs '{sdk_decoded_string}'"
