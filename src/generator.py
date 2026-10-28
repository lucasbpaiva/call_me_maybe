import sys
import numpy as np
from llm_sdk import Small_LLM_Model


def get_masked_argmax(logits: list[float], allowed_ids: list[int]) -> int:
    """Applies a boolean mask to logits and returns the highest allowed token ID."""
    logits_array = np.array(logits)
    mask = np.zeros_like(logits_array, dtype=bool)
    mask[allowed_ids] = True
    logits_array[~mask] = -np.inf
    return int(np.argmax(logits_array))


def generate_sequence(
    model: Small_LLM_Model,
    prompt: str,
    eos_token_id: int,
    allowed_ids: list[int],
    max_length: int = 100
) -> str:
    """
    Generates model completion for a given prompt.

    Generates text token-by-token until the EOS token is reached
    or max_length is exceeded.
    """
    try:
        input_ids: list[int] = model.encode(prompt).tolist()[0]
        generated_count = 0
        while generated_count < max_length:
            logits: list[float] = model.get_logits_from_input_ids(input_ids)
            next_token_id = get_masked_argmax(logits, allowed_ids)
            if next_token_id == eos_token_id:
                break
            input_ids.append(next_token_id)
            generated_count += 1
        return model.decode(input_ids)
    except Exception as err:
        print(f"Error during sequence generation: {err}")
        sys.exit(1)
