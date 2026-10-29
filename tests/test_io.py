import json
import pytest  # type: ignore[import-untyped, unused-ignore]
from pathlib import Path
from src.io_utils import load_function_definitions


def test_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    """Test handling of a non-existent file."""
    with pytest.raises(SystemExit) as exc_info:
        load_function_definitions("non_existent_file.json")
    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Error: File not found" in stderr


def test_empty_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test handling of a completely empty file."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("")
    with pytest.raises(SystemExit) as exc_info:
        load_function_definitions(str(empty_file))
    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Error: File is empty" in stderr


def test_invalid_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test handling of malformed JSON syntax."""
    bad_json_file = tmp_path / "bad.json"
    bad_json_file.write_text("[{unquoted_key: 'value'}]")

    with pytest.raises(SystemExit) as exc_info:
        load_function_definitions(str(bad_json_file))

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Error: Invalid JSON format" in stderr


def test_json_object_instead_of_array(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test schema mismatch when the root is a dict instead of a list."""
    obj_file = tmp_path / "object.json"
    obj_file.write_text(json.dumps({"name": "fn_test"}))

    with pytest.raises(SystemExit) as exc_info:
        load_function_definitions(str(obj_file))

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Expected a JSON array at the root" in stderr


def test_missing_required_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test Pydantic validation when required schema keys are missing."""
    missing_keys_file = tmp_path / "missing_keys.json"
    # Missing 'description', 'parameters', and 'returns'
    missing_keys_file.write_text(json.dumps([{"name": "fn_test"}]))

    with pytest.raises(SystemExit) as exc_info:
        load_function_definitions(str(missing_keys_file))

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Validation Error" in stderr
    assert "Field required" in stderr


def test_unknown_parameter_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test Pydantic validation when a parameter type is unrecognized."""
    unknown_type_file = tmp_path / "unknown_type.json"
    invalid_data = [{
        "name": "fn_test",
        "description": "Test func",
        "parameters": {
            "a": {"type": "magic_type"}  # 'magic_type' is not valid
        },
        "returns": {"type": "number"}
    }]
    unknown_type_file.write_text(json.dumps(invalid_data))
    with pytest.raises(SystemExit) as exc_info:
        load_function_definitions(str(unknown_type_file))
    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Validation Error" in stderr
    assert "magic_type" in stderr
