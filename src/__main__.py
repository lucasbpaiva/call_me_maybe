import argparse
from pathlib import Path
# from llm_sdk import Small_LLM_Model
# from src.generator import generate_sequence
# from src.vocab import build_token_dictionary
from src.io_utils import load_function_definitions, load_input_prompts


def parse_argv() -> argparse.Namespace:
    """Parses command-line arguments for the inference engine.

    Configures the argument parser for executing the JSON-constrained LLM
    inference. Establishes the expected flags and default file paths for
    function schemas, input prompts, and the output record destination.

    Returns:
        argparse.Namespace: An object containing the parsed command-line
        arguments as attributes (functions_definition, input, and output).
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
    args = parse_argv()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    functions = load_function_definitions(args.functions_definition)
    test_cases = load_input_prompts(args.input)
    print(
        f"Loaded {len(functions)} functions and {len(test_cases)} test cases."
    )
    output_path.write_text("test")


if __name__ == "__main__":
    main()
