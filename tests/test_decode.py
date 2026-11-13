from llm_sdk import Small_LLM_Model
from src.tokenizer import Tokenizer


def test_decoder() -> None:
    """Asserts our decode logic exactly matches the SDK's decode output."""
    sdk_model = Small_LLM_Model()
    tokenizer_path = sdk_model.get_path_to_tokenizer_file()
    student_tokenizer = Tokenizer.load_from_file(tokenizer_path)
    # Test matrix covering ASCII, whitespace, emojis, CJK, and ChatML
    test_strings = [
        "Hello world!",
        "12345.6789",
        "-1.79e308",
        "   \t\r\n  ",
        "Café",
        "こんにちは",
        "🔥🚀",
        '{"name": "fn_test", "parameters": {}}',
        "<|im_start|>system\nTest<|im_end|>\n<|im_start|>user\nHi<|im_end|>"
    ]
    for text in test_strings:
        ids = sdk_model.encode(text).tolist()[0]
        sdk_out = sdk_model.decode(ids)
        student_out = student_tokenizer.decode(ids)
        assert student_out == sdk_out, f"Mismatch on text: {text!r}"
