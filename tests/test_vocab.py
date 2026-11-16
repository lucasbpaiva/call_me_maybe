from llm_sdk import Small_LLM_Model
from src.tokenizer import Tokenizer
from src.vocab import build_token_dictionary


def test_vocab_dictionary_matches_sdk() -> None:
    """Tests that the custom token dictionary accurately matches the model SDK.

    Iterates through a sample of 2000 token IDs, verifying that the string
    representation produced by the locally built vocabulary dictionary exactly
    matches the output of the model's native `decode` method.

    Raises:
        AssertionError: If the decoded string from the dictionary does not
            match the SDK's decoded string for any given token ID.
    """
    model = Small_LLM_Model()
    tokenizer = Tokenizer.load_from_file(model.get_path_to_tokenizer_file())
    vocab_dict = build_token_dictionary(tokenizer)[0]
    sample_size = 2000
    for token_id in range(sample_size):
        if token_id in vocab_dict:
            my_decoded_string = vocab_dict[token_id]
            sdk_decoded_string = model.decode([token_id])
            assert my_decoded_string == sdk_decoded_string, \
                f"Mismatch at ID {token_id}: " \
                f"'{my_decoded_string}' vs '{sdk_decoded_string}'"
