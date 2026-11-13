import time
import json
import logging
import argparse
from typing import Any
from pathlib import Path
from src.models import TestCase
from llm_sdk import Small_LLM_Model
from src.constraints import PrefixTree
from src.vocab import build_token_dictionary
from src.generator import generate_function_call
from src.io_utils import load_function_definitions, load_input_prompts
logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s: %(message)s"
)


def parse_argv() -> argparse.Namespace:
    """Parses command-line arguments for the inference engine.

    Configures the argument parser for executing the JSON-constrained LLM
    inference. Establishes the expected flags and default file paths for
    function schemas, input prompts, and the output record destination.

    Returns:
        An object containing the parsed command-line arguments as attributes
        (functions_definition, input, and output).
    """
    parser = argparse.ArgumentParser(
        description="Execute JSON-constrained LLM inference."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing available function schemas."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to an input text file containing a list of prompts."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to save the generated JSON output record."
    )
    return parser.parse_args()


def main() -> None:
    """Executes the constrained JSON generation pipeline.

    Parses command-line arguments, loads function schemas and input prompts,
    initializes the language model and constraint engines (PrefixTree,
    vocabulary), and iterates through the test cases. For each prompt, it
    generates a guaranteed-valid JSON function call, parses it, and writes the
    consolidated results to the specified output file.
    """
    start_time = time.perf_counter()
    args = parse_argv()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    functions = load_function_definitions(args.functions_definition)
    test_cases = load_input_prompts(args.input)
    trie = PrefixTree()
    for fn in functions:
        trie.insert(fn.name)
    model = Small_LLM_Model()
    token_dict, eos_token_id = build_token_dictionary(model)
    results = []
    max_retries = 2
    for i, test_case in enumerate(test_cases):
        for attempt in range(max_retries):
            try:
                current_prompt = test_case.prompt
                if attempt > 0:
                    current_prompt += (
                        "\n\n[System Note: Your previous attempt produced an "
                        "invalid structure. Strictly follow the schema.]"
                    )
                current_case = TestCase(prompt=current_prompt)
                generated_json = generate_function_call(
                    model, current_case, functions,
                    eos_token_id, trie, token_dict
                )
                parsed_call = json.loads(generated_json)
                final_obj = {
                    "prompt": test_case.prompt,
                    "name": parsed_call.get("name"),
                    "parameters": parsed_call.get("parameters", {})
                }
                results.append(final_obj)
                break
            except Exception as err:
                if attempt < max_retries - 1:
                    continue
                logging.error(
                    f"Failed mid-generation on test case {i}.\n"
                    f"Prompt: '{test_case.prompt}'\n"
                    f"Error: {type(err).__name__}: {err}\n"
                )
                # Create a fallback object to ensure the final JSON
                # file still maps 1:1 with the input length.
                error_obj: dict[str, Any] = {
                    "prompt": test_case.prompt,
                    "name": "None",
                    "parameters": {},
                }
                results.append(error_obj)
    formatted_json = json.dumps(results, indent=2)
    output_path.write_text(formatted_json)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Run completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
