from llm_sdk import Small_LLM_Model
from src.tokenizer import PreTokenizer, Tokenizer


def test_pretokenizer_rules() -> None:
    """Verifies Qwen's pre-tokenization regex rules on edge cases.

    Tests contraction splitting, leading space attachment to words, individual
    digit isolation, punctuation with trailing newlines, multi-space run
    splitting before words, and Unicode letter/symbol handling across scripts.
    """
    pt = PreTokenizer()

    # Rule 1: English contractions split as separate pieces
    assert pt.pre_tokenize("don't I'll") == ["don", "'t", " I", "'ll"]

    # Rule 2: Leading space attaches to words
    assert pt.pre_tokenize("Hello world") == ["Hello", " world"]

    # Rule 3: Every digit splits individually
    assert pt.pre_tokenize("123.45") == ["1", "2", "3", ".", "4", "5"]

    # Rule 4 & 5: Punctuation with trailing newlines & standalone newlines
    assert pt.pre_tokenize("Hi!\n\nNext") == ["Hi", "!\n\n", "Next"]

    # Rule 6: Whitespace run keeps last space attached to the following word
    assert pt.pre_tokenize("   hello") == ["  ", " hello"]

    # Unicode letters (\p{L}) and digits (\p{N}) across scripts
    assert pt.pre_tokenize("Café 日本語 🚀") == ["Café", " 日本語", " 🚀"]


def test_pretokenizer_lossless_partition() -> None:
    """Asserts pre-tokenization never drops or alters characters.

    Loads the production tokenizer configuration and verifies that joining all
    pre-tokenized pieces reconstructs the exact original string across empty
    inputs, whitespace runs, JSON payloads, and mixed Unicode scripts.
    """
    model = Small_LLM_Model()
    tok = Tokenizer.load_from_file(model.get_path_to_tokenizer_file())

    samples = [
        "",
        "   \t\r\n  ",
        r'{"name": "fn_test", "parameters": {"x": -42.5}}',
        "Mixed 日本語, emojis 🔥🚀, and contractions we've!",
    ]
    for sample in samples:
        pieces = tok.pre_tokenizer.pre_tokenize(sample)
        assert "".join(pieces) == sample
