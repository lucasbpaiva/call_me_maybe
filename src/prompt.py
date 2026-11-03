import json
from src.models import FunctionDefinition


def build_prompt(
    functions: list[FunctionDefinition],
    user_prompt: str
) -> str:
    """Formats function schemas and user prompt into a raw ChatML string."""
    functions_dict = [fn.model_dump(by_alias=True) for fn in functions]
    functions_json = json.dumps(functions_dict, indent=2)
    system_content = (
        "You are a helpful assistant with access to the following functions:\n"
        f"{functions_json}\n\n"
        "Select the appropriate function to answer the user's request. "
        "You must respond with a JSON object in this exact format: "
        '{"name": "function_name", "parameters": {"param_name": "value"}}'
    )
    # Construct standard ChatML markup ending with the forced JSON prefix
    formatted_prompt = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f'<|im_start|>assistant\n{{"name": "'
    )
    return formatted_prompt
