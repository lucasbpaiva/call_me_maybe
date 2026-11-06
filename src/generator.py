import sys
import numpy as np
from llm_sdk import Small_LLM_Model
from src.prompt import build_prompt
from src.constraints import PrefixTree, NumberValidator, StringValidator
from src.models import TestCase, FunctionDefinition


def get_input_ids(model: Small_LLM_Model, raw_prompt: str) -> list[int]:
    """Tokenizes the raw prompt using the public model API."""
    input_tensor = model.encode(raw_prompt)
    input_ids: list[int] = input_tensor.tolist()[0]
    return input_ids


def get_allowed_token_ids(
    trie: PrefixTree,
    current_prefix: str,
    token_dict: dict[int, str]
) -> list[int]:
    """Evaluates the vocabulary against the Trie to find valid next tokens."""
    allowed_ids: list[int] = []
    is_complete_name = trie.search(current_prefix)
    for token_id, token_str in token_dict.items():
        if is_complete_name and token_str.startswith('"'):
            allowed_ids.append(token_id)
            continue
        candidate_string = current_prefix + token_str
        if trie.is_prefix(candidate_string):
            allowed_ids.append(token_id)
    return allowed_ids


def get_allowed_number_tokens(
    validator: NumberValidator,
    current_segment: str,
    token_dict: dict[int, str],
    is_last_parameter: bool
) -> list[int]:
    """Finds valid next tokens for a number or its termination delimiter."""
    allowed_ids: list[int] = []
    is_currently_complete = validator.is_complete(current_segment)
    closing_delimiter = "}" if is_last_parameter else ","
    for token_id, token_str in token_dict.items():
        candidate_string = current_segment + token_str
        if validator.is_partial(candidate_string):
            allowed_ids.append(token_id)
            continue
        if is_currently_complete and token_str.startswith(closing_delimiter):
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
) -> str:
    """Generates and constrains the function name segment of the JSON output.

    Utilizes a PrefixTree to mask model logits, ensuring the generated sequence
    strictly follows a valid function name path. The generation loop appends
    valid tokens to the provided context and terminates when the model outputs
    a closing quote ('"').

    Args:
        input_ids: The current sequence of context token IDs. Newly
            generated token IDs are appended to this list in-place.
        trie: The Trie data structure containing the allowed
            function names.
        token_dict: A mapping of token IDs to their
            corresponding string representations in the model's vocabulary.
        eos_token_id: The token ID representing the end of a sequence.
        model: The language model instance used for inference.
        max_length: The maximum number of tokens to generate
            for this segment. Defaults to 100.

    Returns:
        The fully generated valid function name (without the closing quote).
    """
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
            token_str = token_dict[next_token_id]
            if '"' in token_str:
                break
            input_ids.append(next_token_id)
            current_segment += token_str
            generated_count += 1
        return current_segment
    except Exception as err:
        print(f"Error during sequence generation: {err}")
        sys.exit(1)


def generate_number_parameter(
    input_ids: list[int],
    model: Small_LLM_Model,
    token_dict: dict[int, str],
    eos_token_id: int,
    is_last_parameter: bool
) -> None:
    """Generates a constrained numeric value for a JSON parameter.

    Uses a state machine to evaluate and mask model logits, forcing the
    LLM to output only tokens that form a strictly valid JSON number. The
    generation loop terminates when the model outputs the appropriate closing
    delimiter (either a comma or a closing brace), leaving the delimiter
    unappended so the orchestrator can cleanly inject it.

    Args:
        input_ids: The current sequence of token IDs representing the context.
            Tokens generated during this phase are appended in-place.
        model: The language model instance used for inference.
        token_dict: A mapping of token IDs to their corresponding
            string representations from the model's vocabulary.
        eos_token_id: The token ID representing the end of a sequence.
        is_last_parameter: Indicates if this is the final parameter in
            the schema. Determines whether the termination delimiter should be
            a closing brace ('}') or a comma (',').
    """
    validator = NumberValidator()
    current_segment = ""
    closing_delimiter = "}" if is_last_parameter else ","
    generated_token_count = 0
    while True:
        args = validator, current_segment, token_dict, is_last_parameter
        allowed_ids = get_allowed_number_tokens(*args)
        if not allowed_ids:
            break
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = get_masked_argmax(logits, allowed_ids)
        if next_token_id == eos_token_id:
            break
        token_str = token_dict[next_token_id]
        if closing_delimiter in token_str:
            break
        input_ids.append(next_token_id)
        current_segment += token_str
        generated_token_count += 1
    if generated_token_count > 0:
        del input_ids[-generated_token_count:]
    normalized_str = str(float(current_segment))
    normalized_ids = get_input_ids(model, normalized_str)
    input_ids.extend(normalized_ids)


def find_unescaped_quote(text: str) -> int:
    """Finds the index of the first unescaped double quote."""
    escaped = False
    for i, char in enumerate(text):
        if char == '\\' and not escaped:
            escaped = True
        elif char == '"' and not escaped:
            return i
        else:
            escaped = False
    return -1


def get_allowed_string_tokens(
    validator: StringValidator,
    current_segment: str,
    token_dict: dict[int, str]
) -> list[int]:
    """Finds valid next tokens for a string, allowing a terminating quote."""
    allowed_ids = []
    for token_id, token_str in token_dict.items():
        candidate = current_segment + token_str
        quote_idx = find_unescaped_quote(candidate)
        if quote_idx != -1:
            payload = candidate[:quote_idx]
            if validator.is_partial(payload):
                allowed_ids.append(token_id)
        else:
            if validator.is_partial(candidate):
                allowed_ids.append(token_id)
    return allowed_ids


def generate_string_parameter(
    input_ids: list[int],
    model: Small_LLM_Model,
    token_dict: dict[int, str],
    eos_token_id: int
) -> None:
    """Generates a JSON string parameter and appends it to the context."""
    validator = StringValidator()
    current_segment = ""
    generated_token_count = 0
    input_ids.extend(get_input_ids(model, '"'))
    while True:
        args = validator, current_segment, token_dict
        allowed_ids = get_allowed_string_tokens(*args)
        if not allowed_ids:
            break
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = get_masked_argmax(logits, allowed_ids)
        if next_token_id == eos_token_id:
            break
        token_str = token_dict[next_token_id]
        candidate = current_segment + token_str
        quote_idx = find_unescaped_quote(candidate)
        if quote_idx != -1:
            valid_payload = candidate[:quote_idx]
            added_text = valid_payload[len(current_segment):]
            current_segment += added_text
            break
        input_ids.append(next_token_id)
        current_segment += token_str
        generated_token_count += 1
    if generated_token_count > 0:
        del input_ids[-generated_token_count:]
    clean_ids = get_input_ids(model, current_segment)
    input_ids.extend(clean_ids)
    input_ids.extend(get_input_ids(model, '"'))


def get_allowed_boolean_tokens(
    trie: PrefixTree,
    current_segment: str,
    token_dict: dict[int, str],
    is_last_parameter: bool
) -> list[int]:
    """Finds valid next tokens for a boolean value."""
    allowed_ids = []
    is_complete = trie.search(current_segment)
    closing_delimiter = "}" if is_last_parameter else ","
    for token_id, token_str in token_dict.items():
        candidate = current_segment + token_str
        if trie.is_prefix(candidate):
            allowed_ids.append(token_id)
            continue
        if is_complete and token_str.startswith(closing_delimiter):
            allowed_ids.append(token_id)
    return allowed_ids


def generate_boolean_parameter(
    input_ids: list[int],
    model: Small_LLM_Model,
    token_dict: dict[int, str],
    eos_token_id: int,
    is_last_parameter: bool
) -> None:
    """Generates a constrained boolean ('true' or 'false') and appends it."""
    boolean_trie = PrefixTree()
    boolean_trie.insert("true")
    boolean_trie.insert("false")
    current_segment = ""
    closing_delimiter = "}" if is_last_parameter else ","
    generated_token_count = 0
    while True:
        args = boolean_trie, current_segment, token_dict, is_last_parameter
        allowed_ids = get_allowed_boolean_tokens(*args)
        if not allowed_ids:
            break
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = get_masked_argmax(logits, allowed_ids)
        if next_token_id == eos_token_id:
            break
        token_str = token_dict[next_token_id]
        if closing_delimiter in token_str:
            break
        input_ids.append(next_token_id)
        current_segment += token_str
        generated_token_count += 1
    if generated_token_count > 0:
        del input_ids[-generated_token_count:]
    clean_ids = get_input_ids(model, current_segment)
    input_ids.extend(clean_ids)


def generate_parameters(
    model: Small_LLM_Model,
    input_ids: list[int],
    schema: FunctionDefinition,
    token_dict: dict[int, str],
    eos_token_id: int
) -> None:
    """Orchestrates parameter generation based on the function schema."""
    parameters = schema.parameters if schema.parameters else {}
    param_items = list(parameters.items())
    if not param_items:
        input_ids.extend(get_input_ids(model, "}"))
        return
    for i, (param_name, param_details) in enumerate(param_items):
        is_last_parameter = (i == len(param_items) - 1)
        key_text = f'"{param_name}": '
        input_ids.extend(get_input_ids(model, key_text))
        param_type = param_details.arg_type
        if param_type == "number":
            a_0 = input_ids, model, token_dict, eos_token_id, is_last_parameter
            generate_number_parameter(*a_0)
        elif param_type == "string":
            a_1 = input_ids, model, token_dict, eos_token_id
            generate_string_parameter(*a_1)
        elif param_type == "boolean":
            a_2 = input_ids, model, token_dict, eos_token_id, is_last_parameter
            generate_boolean_parameter(*a_2)
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
        delimiter = "}" if is_last_parameter else ", "
        input_ids.extend(get_input_ids(model, delimiter))


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
    generated_start = len(input_ids)
    args_1 = input_ids, trie, token_dict, eos_token_id, model
    selected_name = select_function_name(*args_1)
    selected_schema = next(fn for fn in functions if fn.name == selected_name)
    transition_ids = get_input_ids(model, '", "parameters": {')
    input_ids.extend(transition_ids)
    args_2 = model, input_ids, selected_schema, token_dict, eos_token_id
    generate_parameters(*args_2)
    generated = model.decode(input_ids[generated_start:])
    return '{"name": "' + generated + '}'
