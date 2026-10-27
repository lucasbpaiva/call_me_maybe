from llm_sdk import Small_LLM_Model
from src.generator import generate_sequence
from src.vocab import build_token_dictionary


def main() -> None:
    model = Small_LLM_Model()
    prompt = "The capital of Brazil is"
    eos_token_id = build_token_dictionary(model)[1]
    print(generate_sequence(model, prompt, eos_token_id))


if __name__ == "__main__":
    main()
