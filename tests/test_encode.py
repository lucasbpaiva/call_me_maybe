import json
from pathlib import Path
import pytest
from llm_sdk import Small_LLM_Model
from src.tokenizer import AddedToken, SpecialTokens, Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Initializes a shared Tokenizer instance from the SDK tokenizer file.

    Returns:
        A validated Tokenizer loaded from the model's tokenizer.json path.
    """
    model = Small_LLM_Model()
    return Tokenizer.load_from_file(model.get_path_to_tokenizer_file())


def test_special_tokens_longest_first() -> None:
    """Verifies longer added tokens match before shorter overlapping prefixes.

    Configures a SpecialTokens instance with two tokens sharing a common
    prefix and asserts that splitting prioritizes the longest matching token
    over partial prefix matches.
    """
    st = SpecialTokens(
        tokens=[
            AddedToken(id=1, content="<|im|>", special=True),
            AddedToken(id=2, content="<|im_start|>", special=True),
        ]
    )
    assert st.split("hi<|im_start|>user") == [
        ("hi", False),
        ("<|im_start|>", True),
        ("user", False),
    ]


def test_encode_against_golden(tokenizer: Tokenizer) -> None:
    """Verifies Tokenizer.encode matches SDK token IDs across all golden cases.

    Loads reference text strings and expected token IDs from tests/golden.json,
    skipping partial-prefix decode records, and asserts exact ID sequence
    equality between the custom encoder and the SDK oracle.

    Args:
        tokenizer: The initialized Tokenizer fixture under test.
    """
    cases = json.loads(Path("tests/golden.json").read_text(encoding="utf-8"))
    for case in cases:
        if case["text"].startswith("<PREFIX "):
            continue
        actual_ids = tokenizer.encode(case["text"])
        assert actual_ids == case["ids"], (
            f"Encode mismatch on {case['text']!r}\n"
            f"  Expected: {case['ids']}\n"
            f"  Actual  : {actual_ids}"
        )
