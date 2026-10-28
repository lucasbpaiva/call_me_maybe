from llm_sdk import Small_LLM_Model
from src.generator import generate_sequence
from src.vocab import build_token_dictionary


# When generating the next token, pass this list to your mask
# next_token_id = get_masked_argmax(logits, digit_token_ids)
def main() -> None:
    model = Small_LLM_Model()
    prompt = "What is 2 + 3? "
    vocab_dict, eos_token_id = build_token_dictionary(model)

    # Find all tokens that are strictly numeric (e.g., "1", "42", "2026")
    digit_token_ids = [
        token_id for token_id, token_str in vocab_dict.items()
        if token_str.isdigit()
    ]

    # target_words = ["Yes", "No", "Maybe"]
    # allowed_word_ids = []
    # for word in target_words:
    #     # model.encode() returns a 2D tensor, extract the first token integer
    #     token_id = model.encode(word).tolist()[0][0]
    #     allowed_word_ids.append(token_id)

    print(generate_sequence(model, prompt, eos_token_id, digit_token_ids, 1))


if __name__ == "__main__":
    main()
