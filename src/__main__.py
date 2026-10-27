import sys
import numpy as np
from llm_sdk import Small_LLM_Model


def main() -> None:
    print(
        "Loading model "
        "(this may take a moment to download weights on the first run)..."
    )
    # 1. Safe Initialization
    try:
        model = Small_LLM_Model()
    except Exception as err:
        print(f"Critical error initializing the model: {err}")
        sys.exit(1)
    prompt = input("Your prompt: ")
    # 2. Encode and isolate the forbidden PyTorch Tensor
    try:
        # .encode returns a 2D tensor on the target device
        tensor_ids = model.encode(prompt)
        # Convert to a 1D Python list: [[id1, id2]] -> [id1, id2]
        input_ids: list[int] = tensor_ids.tolist()[0]
    except Exception as err:
        print(f"Error encoding prompt: {err}")
        sys.exit(1)
    print(f"\nInitial prompt: '{prompt}'")
    print(f"Initial token IDs: {input_ids}")
    # 3. The Token-by-Token Generation Loop
    max_tokens = 5
    for step in range(max_tokens):
        try:
            # Retrieve raw logits for next token based on the sequence so far
            logits: list[float] = model.get_logits_from_input_ids(input_ids)
            # Find the index of the highest logit using numpy
            next_token_id = int(np.argmax(logits))
            # Append the predicted token to the sequence
            input_ids.append(next_token_id)
            print(f"Step {step + 1}: Model predicted token ID {next_token_id}")
        except Exception as err:
            print(f"Error during forward pass at step {step}: {err}")
            sys.exit(1)
    # 4. Decode the final sequence back to a string
    try:
        final_text: str = model.decode(input_ids)
        print(f"\nFinal generated text:\n{final_text}")
    except Exception as err:
        print(f"Error decoding output: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
