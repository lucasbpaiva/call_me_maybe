import sys
import numpy as np
from llm_sdk import Small_LLM_Model
from src.prompt import build_prompt
from src.constraints import PrefixTree
from src.models import TestCase, FunctionDefinition


def get_input_ids(model: Small_LLM_Model, raw_prompt: str) -> list[int]:
    """Tokenizes the raw prompt using the public model API."""
    input_tensor = model.encode(raw_prompt)
    return input_tensor.tolist()[0]


def get_allowed_token_ids(
    trie: PrefixTree,
    current_prefix: str,
    token_dict: dict[int, str]
) -> list[int]:
    """Evaluates the vocabulary against the Trie to find valid next tokens."""
    allowed_ids = []
    for token_id, token_str in token_dict.items():
        candidate_string = current_prefix + token_str
        if trie.is_prefix(candidate_string):
            allowed_ids.append(token_id)
    return allowed_ids


def get_masked_argmax(logits: list[float], allowed_ids: list[int]) -> int:
    """Applies a boolean mask and returns the highest allowed token ID."""
    logits_array = np.array(logits)
    mask = np.zeros_like(logits_array, dtype=bool)
    mask[allowed_ids] = True
    logits_array[~mask] = -np.inf
    return int(np.argmax(logits_array))


def select_function_name(
    input_ids: list[int],
    trie: PrefixTree,
    token_dict: dict[int, str],
    eos_token_id: int,
    model: Small_LLM_Model,
    max_length: int = 100
):
    try:
        generated_count = 0
        current_segment = ""
        while generated_count < max_length:
            allowed_ids = get_allowed_token_ids(
                trie, current_segment, token_dict
            )
            if not allowed_ids:
                break
            logits: list[float] = model.get_logits_from_input_ids(input_ids)
            next_token_id = get_masked_argmax(logits, allowed_ids)
            if next_token_id == eos_token_id:
                break
            input_ids.append(next_token_id)
            current_segment += token_dict[next_token_id]
            generated_count += 1
            if trie.search(current_segment):
                break
        return current_segment
    except Exception as err:
        print(f"Error during sequence generation: {err}")
        sys.exit(1)


def generate_function_call(
    model: Small_LLM_Model,
    test_case: TestCase,
    functions: list[FunctionDefinition],
    eos_token_id: int,
    trie: PrefixTree,
    token_dict: dict[int, str],
    max_length: int = 100
) -> str:
    """Executes the forced-JSON generation phases for a single prompt."""
    prompt = build_prompt(functions, test_case.prompt)
    input_ids: list[int] = get_input_ids(model, prompt)
    selected_name = select_function_name(
        input_ids, trie, token_dict, eos_token_id, model
    )
    selected_schema = next(fn for fn in functions if fn.name == selected_name)
    transition_ids = get_input_ids(model, '", "parameters": {}')
    input_ids.extend(transition_ids)
    # 4. Hand off to the new parameter constraint engine
    # generate_parameters(model, input_ids, selected_schema, token_dict)
    full_output = model.decode(input_ids) + "}"
    json_start_index = full_output.rfind('{"name": "')
    return full_output[json_start_index:]
