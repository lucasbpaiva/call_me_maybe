import json
from pathlib import Path
import pytest
from llm_sdk import Small_LLM_Model
from src.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Initializes a shared Tokenizer instance from the SDK tokenizer file.

    Returns:
        A validated Tokenizer loaded from the model's tokenizer.json path.
    """
    model = Small_LLM_Model()
    return Tokenizer.load_from_file(model.get_path_to_tokenizer_file())


def test_decode_against_golden(tokenizer: Tokenizer) -> None:
    """Verifies that Tokenizer.decode matches the SDK oracle across all cases.

    Loads pre-recorded token ID sequences and expected decoded strings from
    tests/golden.json, asserting exact string equality when decoding with
    special tokens skipped.

    Args:
        tokenizer: The initialized Tokenizer fixture under test.
    """
    golden_path = Path("tests/golden.json")
    cases = json.loads(golden_path.read_text(encoding="utf-8"))

    for case in cases:
        result = tokenizer.decode(case["ids"], skip_special_tokens=True)
        assert result == case["decoded"], (
            f"Mismatch on {case['text']!r}\n"
            f"  IDs     : {case['ids']}\n"
            f"  Expected: {case['decoded']!r}\n"
            f"  Actual  : {result!r}"
        )


def test_decode_unknown_id_raises(tokenizer: Tokenizer) -> None:
    """Asserts that decoding an out-of-vocabulary token ID raises ValueError.

    Args:
        tokenizer: The initialized Tokenizer fixture under test.
    """
    with pytest.raises(ValueError, match="Unknown token ID"):
        tokenizer.decode([999999])
