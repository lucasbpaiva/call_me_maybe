import numpy as np
from typing import Callable
from llm_sdk import Small_LLM_Model
from src.tokenizer import Tokenizer
from src.prompt import build_prompt
from src.constraints import PrefixTree, NumberValidator, StringValidator
from src.models import TestCase, FunctionDefinition


def make_name_predicate(
    trie: PrefixTree, current_prefix: str
) -> Callable[[str], bool]:
    """Creates a predicate to validate tokens for the function name segment.

    Args:
        trie: The Trie containing allowed function names.
        current_prefix: The function name prefix generated so far.

    Returns:
        A function that accepts a token string and returns True if appending
        it forms a valid prefix or a correctly terminated name.
    """
    is_complete_name = trie.search(current_prefix)

    def is_allowed(token_str: str) -> bool:
        if is_complete_name and token_str.startswith('"'):
            return True
        return trie.is_prefix(current_prefix + token_str)

    return is_allowed


def make_number_predicate(
    validator: NumberValidator,
    current_segment: str,
    is_last_parameter: bool
) -> Callable[[str], bool]:
    """Creates a predicate to validate tokens for a JSON number parameter.

    Args:
        validator: The state machine enforcing JSON number syntax.
        current_segment: The number string generated so far.
        is_last_parameter: Indicates if the delimiter should be a brace ('}')
            rather than a comma (',').

    Returns:
        A function that returns True if the token maintains a valid partial
        number or legally terminates a complete number.
    """
    is_currently_complete = validator.is_complete(current_segment)
    closing_delimiter = "}" if is_last_parameter else ","

    def is_allowed(token_str: str) -> bool:
        candidate = current_segment + token_str
        if validator.is_partial(candidate):
            return True
        if is_currently_complete and token_str.startswith(closing_delimiter):
            return True
        return False

    return is_allowed


def make_string_predicate(
    validator: StringValidator,
    current_segment: str
) -> Callable[[str], bool]:
    """Creates a predicate to validate tokens for a JSON string parameter.

    Args:
        validator: The state machine enforcing JSON string escapes.
        current_segment: The string payload generated so far.

    Returns:
        A function that returns True if the token maintains valid string
        formatting, even if it contains a mid-token terminating quote.
    """
    def is_allowed(token_str: str) -> bool:
        candidate = current_segment + token_str
        quote_idx = find_unescaped_quote(candidate)
        if quote_idx != -1:
            payload = candidate[:quote_idx]
            return validator.is_partial(payload)
        return validator.is_partial(candidate)

    return is_allowed


def make_boolean_predicate(
    trie: PrefixTree,
    current_segment: str,
    is_last_parameter: bool
) -> Callable[[str], bool]:
    """Creates a predicate to validate tokens for a JSON boolean parameter.

    Args:
        trie: The Trie containing the exact literals 'true' and 'false'.
        current_segment: The boolean sequence generated so far.
        is_last_parameter: Indicates if the delimiter should be a brace ('}')
            rather than a comma (',').

    Returns:
        A function that returns True if the token builds towards
        a valid boolean literal or safely terminates a completed one.
    """
    is_complete = trie.search(current_segment)
    closing_delimiter = "}" if is_last_parameter else ","

    def is_allowed(token_str: str) -> bool:
        candidate = current_segment + token_str
        if trie.is_prefix(candidate):
            return True
        if is_complete and token_str.startswith(closing_delimiter):
            return True
        return False

    return is_allowed


def pick_best_allowed(
    logits: list[float],
    token_dict: dict[int, str],
    is_allowed: Callable[[str], bool]
) -> int:
    """Sorts and evaluates logits to find the highest-scoring valid token.

    Sorts the model's output logits to lazily evaluate candidate tokens against
    the provided predicate. This prevents scanning the entire vocabulary,
    significantly improving performance by short-circuiting as soon as a valid
    token is found. Empty string tokens are automatically ignored to prevent
    infinite generation loops.

    Args:
        logits: The raw logits or probabilities over the vocabulary.
        token_dict: A mapping of token IDs to their string representations.
        is_allowed: A predicate function that takes a candidate token string
            and returns True if it maintains valid syntax.

    Returns:
        The token ID of the highest-scoring valid candidate.

    Raises:
        ValueError: If no token in the vocabulary satisfies the predicate.
    """
    scores = np.asarray(logits, dtype=np.float32)
    sorted_indices = np.argsort(-scores)
    for token_id in sorted_indices:
        token_str = token_dict.get(int(token_id))
        if not token_str:
            continue
        if is_allowed(token_str):
            return int(token_id)
    raise ValueError("No valid token found in the vocabulary.")


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
    generated_count = 0
    current_segment = ""
    while generated_count < max_length:
        logits: list[float] = model.get_logits_from_input_ids(input_ids)
        is_allowed = make_name_predicate(trie, current_segment)
        next_token_id = pick_best_allowed(logits, token_dict, is_allowed)
        if next_token_id == eos_token_id:
            break
        token_str = token_dict[next_token_id]
        if '"' in token_str:
            break
        input_ids.append(next_token_id)
        current_segment += token_str
        generated_count += 1
    return current_segment


def generate_number_parameter(
    input_ids: list[int],
    model: Small_LLM_Model,
    tokenizer: Tokenizer,
    token_dict: dict[int, str],
    eos_token_id: int,
    is_last_parameter: bool,
    max_length: int = 100
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
        tokenizer: The custom Tokenizer instance used for encoding.
        token_dict: A mapping of token IDs to their corresponding
            string representations from the model's vocabulary.
        eos_token_id: The token ID representing the end of a sequence.
        is_last_parameter: Indicates if this is the final parameter in
            the schema. Determines whether the termination delimiter should be
            a closing brace ('}') or a comma (',').
        max_length: The maximum number of tokens to generate for this segment.
            Defaults to 100.
    """
    validator = NumberValidator()
    current_segment = ""
    closing_delimiter = "}" if is_last_parameter else ","
    generated_token_count = 0
    while generated_token_count < max_length:
        logits = model.get_logits_from_input_ids(input_ids)
        args = validator, current_segment, is_last_parameter
        is_allowed = make_number_predicate(*args)
        next_token_id = pick_best_allowed(logits, token_dict, is_allowed)
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
    normalized_ids = tokenizer.encode(normalized_str)
    input_ids.extend(normalized_ids)


def find_unescaped_quote(text: str) -> int:
    """Finds the index of the first unescaped double quote in a string.

    Iterates through the string evaluating backslash escape sequences to
    differentiate between literal quotes inside a JSON string and the
    structural quote that terminates the string.

    Args:
        text: The string to search through.

    Returns:
        The integer index of the first unescaped double quote, or -1 if
        no unescaped quote is found.
    """
    escaped = False
    for i, char in enumerate(text):
        if char == '\\' and not escaped:
            escaped = True
        elif char == '"' and not escaped:
            return i
        else:
            escaped = False
    return -1


def generate_string_parameter(
    input_ids: list[int],
    model: Small_LLM_Model,
    tokenizer: Tokenizer,
    token_dict: dict[int, str],
    eos_token_id: int,
    max_length: int = 100
) -> None:
    """Generates a constrained string value for a JSON parameter.

    Uses a state machine to evaluate and mask model logits, forcing the
    LLM to output only tokens that form a valid JSON string with correct
    escaping. The generation loop terminates when an unescaped double
    quote is produced.

    Args:
        input_ids: The current sequence of token IDs representing the context.
        model: The language model instance used for inference.
        tokenizer: The custom Tokenizer instance used for encoding.
        token_dict: A mapping of token IDs to their corresponding string
            representations from the model's vocabulary.
        eos_token_id: The token ID representing the end of a sequence.
        max_length: The maximum number of tokens to generate for this segment.
            Defaults to 100.
    """
    validator = StringValidator()
    current_segment = ""
    generated_token_count = 0
    input_ids.extend(tokenizer.encode('"'))
    while generated_token_count < max_length:
        logits = model.get_logits_from_input_ids(input_ids)
        is_allowed = make_string_predicate(validator, current_segment)
        next_token_id = pick_best_allowed(logits, token_dict, is_allowed)
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
    clean_ids = tokenizer.encode(current_segment)
    input_ids.extend(clean_ids)
    input_ids.extend(tokenizer.encode('"'))


def generate_boolean_parameter(
    input_ids: list[int],
    model: Small_LLM_Model,
    tokenizer: Tokenizer,
    token_dict: dict[int, str],
    eos_token_id: int,
    is_last_parameter: bool,
    max_length: int = 100
) -> None:
    """Generates a constrained boolean value ('true' or 'false').

    Uses a local PrefixTree containing only 'true' and 'false' to evaluate
    model logits. The generation loop terminates when the model outputs
    the appropriate closing delimiter.

    Args:
        input_ids: The current sequence of token IDs representing the context.
        model: The language model instance used for inference.
        tokenizer: The custom Tokenizer instance used for encoding.
        token_dict: A mapping of token IDs to their corresponding string
            representations from the model's vocabulary.
        eos_token_id: The token ID representing the end of a sequence.
        is_last_parameter: Indicates if this is the final parameter in
            the schema. Determines whether the termination delimiter should be
            a closing brace ('}') or a comma (',').
        max_length: The maximum number of tokens to generate for this segment.
            Defaults to 100.
    """
    boolean_trie = PrefixTree()
    boolean_trie.insert("true")
    boolean_trie.insert("false")
    current_segment = ""
    closing_delimiter = "}" if is_last_parameter else ","
    generated_token_count = 0
    while generated_token_count < max_length:
        logits = model.get_logits_from_input_ids(input_ids)
        args = boolean_trie, current_segment, is_last_parameter
        is_allowed = make_boolean_predicate(*args)
        next_token_id = pick_best_allowed(logits, token_dict, is_allowed)
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
    clean_ids = tokenizer.encode(current_segment)
    input_ids.extend(clean_ids)


def generate_parameters(
    model: Small_LLM_Model,
    tokenizer: Tokenizer,
    input_ids: list[int],
    schema: FunctionDefinition,
    token_dict: dict[int, str],
    eos_token_id: int
) -> None:
    """Orchestrates parameter generation based on the function schema.

    Iterates through the required parameters of the selected function schema,
    injecting the appropriate JSON keys and delegating the value generation
    to the type-specific generator loops (string, number, or boolean).

    Args:
        model: The language model instance used for inference.
        tokenizer: The custom Tokenizer instance used for encoding.
        input_ids: The current sequence of context token IDs.
        schema: The definition of the selected function,
            including its expected parameters.
        token_dict: A mapping of token IDs to their corresponding
            string representations.
        eos_token_id: The token ID representing the end of a sequence.

    Raises:
        ValueError: If the schema defines a parameter type not supported.
    """
    parameters = schema.parameters if schema.parameters else {}
    param_items = list(parameters.items())
    if not param_items:
        input_ids.extend(tokenizer.encode("}"))
        return
    for i, (param_name, param_details) in enumerate(param_items):
        is_last_parameter = (i == len(param_items) - 1)
        key_text = f'"{param_name}": '
        input_ids.extend(tokenizer.encode(key_text))
        param_type = param_details.arg_type
        if param_type == "number":
            generate_number_parameter(
                input_ids, model, tokenizer, token_dict,
                eos_token_id, is_last_parameter
            )
        elif param_type == "string":
            generate_string_parameter(
                input_ids, model, tokenizer, token_dict, eos_token_id
            )
        elif param_type == "boolean":
            generate_boolean_parameter(
                input_ids, model, tokenizer, token_dict,
                eos_token_id, is_last_parameter
            )
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
        delimiter = "}" if is_last_parameter else ", "
        input_ids.extend(tokenizer.encode(delimiter))


def generate_function_call(
    model: Small_LLM_Model,
    tokenizer: Tokenizer,
    test_case: TestCase,
    functions: list[FunctionDefinition],
    eos_token_id: int,
    trie: PrefixTree,
    token_dict: dict[int, str],
) -> str:
    """Executes the forced-JSON generation phases for a single prompt.

    Constructs the prompt, forces the opening JSON syntax into the context,
    and sequentially drives the constrained generation of the function name
    and its parameters. Slices the raw model output to return only the
    cleanly formatted JSON object.

    Args:
        model: The language model instance used for inference.
        test_case: The input test case containing the user prompt.
        functions: A list of all available function definitions.
        eos_token_id: The token ID representing the end of a sequence.
        trie: The PrefixTree populated with all valid function names.
        token_dict: A mapping of token IDs to their corresponding
            string representations.

    Returns:
        The final generated JSON string representing the function call.
    """
    prompt = build_prompt(functions, test_case.prompt)
    input_ids: list[int] = tokenizer.encode(prompt)
    generated_start = len(input_ids)
    args_1 = input_ids, trie, token_dict, eos_token_id, model
    selected_name = select_function_name(*args_1)
    selected_schema = next(fn for fn in functions if fn.name == selected_name)
    transition_ids = tokenizer.encode('", "parameters": {')
    input_ids.extend(transition_ids)
    generate_parameters(
        model, tokenizer, input_ids, selected_schema, token_dict, eos_token_id
    )
    generated = tokenizer.decode(input_ids[generated_start:])
    return '{"name": "' + generated + '}'
