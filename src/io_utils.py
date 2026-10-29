import sys
import json
from pathlib import Path
from pydantic import ValidationError
from src.models import FunctionDefinition, TestCase


def load_function_definitions(filename: str) -> list[FunctionDefinition]:
    """Loads and validates function definitions from a JSON file.

    Reads the specified JSON file, parses its contents, and validates the
    structure against the FunctionDefinition schema. Ensures the file exists,
    is not empty, contains valid JSON, and has a JSON array at its root.
    If any validation step fails, an error is written to standard error and the
    program exits cleanly to prevent downstream crashes.

    Args:
        filename (str): The file path to the JSON document to be loaded.

    Returns:
        list[FunctionDefinition]: A list of validated function definitions.

    Raises:
        SystemExit: If the file cannot be found, is empty, contains malformed
            JSON, does not have an array root, or fails Pydantic schema
            validation.
    """
    file_path = Path(filename)
    if not file_path.is_file():
        sys.stderr.write("Error: File not found\n")
        sys.exit(1)
    content = file_path.read_text(encoding="utf-8")
    if not content.strip():
        sys.stderr.write("Error: File is empty\n")
        sys.exit(1)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        sys.stderr.write("Error: Invalid JSON format\n")
        sys.exit(1)
    if not isinstance(data, list):
        sys.stderr.write("Expected a JSON array at the root\n")
        sys.exit(1)
    validated_functions = []
    try:
        for item in data:
            validated_functions.append(FunctionDefinition(**item))
    except ValidationError as e:
        sys.stderr.write(f"Validation Error: {e}\n")
        sys.exit(1)
    return validated_functions


def load_input_prompts(filename: str) -> list[TestCase]:
    """Loads and validates user prompts from a JSON file.

    Reads the specified JSON file, parses its contents, and validates the
    structure against the TestCase schema. Ensures the file exists, is not
    empty, contains valid JSON, and has a JSON array at its root. If any
    validation step fails, an error is written to standard error and the
    program exits cleanly to prevent downstream crashes.

    Args:
        filename (str): The file path to the JSON document containing the
            input prompts.

    Returns:
        list[TestCase]: A list of strictly validated test cases.

    Raises:
        SystemExit: If the file cannot be found, is empty, contains malformed
            JSON, does not have an array root, or fails Pydantic validation.
    """
    file_path = Path(filename)
    if not file_path.is_file():
        sys.stderr.write(f"Error: Input file '{filename}' not found\n")
        sys.exit(1)
    content = file_path.read_text(encoding="utf-8")
    if not content.strip():
        sys.stderr.write(f"Error: Input file '{filename}' is empty\n")
        sys.exit(1)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        sys.stderr.write(f"Error: Invalid JSON format in '{filename}'\n")
        sys.exit(1)
    if not isinstance(data, list):
        sys.stderr.write("Expected a JSON array at the root'\n")
        sys.exit(1)
    validated_prompts = []
    try:
        for item in data:
            validated_prompts.append(TestCase(**item))
    except ValidationError as e:
        sys.stderr.write(f"Validation Error:\n{e}\n")
        sys.exit(1)
    return validated_prompts
