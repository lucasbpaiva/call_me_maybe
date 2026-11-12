import json
from src.models import FunctionDefinition


def build_prompt(
    functions: list[FunctionDefinition],
    user_prompt: str
) -> str:
    """Formats function schemas and a user prompt into a raw ChatML string.

    Serializes the provided function definitions into a compact JSON string,
    specifically excluding the 'returns' field from the output dictionary.
    It constructs a system message instructing the model to output a strictly
    formatted JSON object, then builds the final sequence using standard ChatML
    special tokens. The prompt deliberately ends by opening the assistant reply
    with the start of a JSON object to force constrained generation.

    Args:
        functions: A list of available function definitions to be injected
            into the system prompt.
        user_prompt: The raw instruction or query from the user.

    Returns:
        The complete ChatML formatted prompt string, terminating precisely
        at the opening quote of the function name key.
    """
    functions_dict = [
        fn.model_dump(
            by_alias=True, exclude_none=True, exclude={"returns"}
        ) for fn in functions
    ]
    functions_json = json.dumps(functions_dict, separators=(",", ":"))
    system_content = (
        "You are a helpful assistant with access to the following functions:\n"
        f"{functions_json}\n\n"
        "Select the appropriate function to answer the user's request. "
        "You must respond with a JSON object in this exact format: "
        r'{"name": "function_name", "parameters": {"param_name": "value"}}'
    )
    formatted_prompt = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        '<|im_start|>assistant\n' + r'{{"name": "'
    )
    return formatted_prompt
