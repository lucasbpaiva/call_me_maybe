import random
from llm_sdk import Small_LLM_Model
from src.tokenizer import Tokenizer


def test_fuzz() -> None:
    """Verifies encode/decode parity against SDK over seeded random text."""
    rng = random.Random(42)
    model = Small_LLM_Model()
    tok = Tokenizer.load_from_file(model.get_path_to_tokenizer_file())
    alphabet = (
        list("abcxyzABCXYZ0123456789 \t\n\r!@#$%^&*()_+-=[]{}|;':\",./<>?")
        + ["é", "e\u0301", "日", "本", "語", "🚀", "🔥", "<|im_start|>", "<think>"]
    )

    for _ in range(200):
        length = rng.randint(1, 60)
        text = "".join(rng.choices(alphabet, k=length))
        expected_ids: list[int] = model.encode(text).tolist()[0]
        actual_ids = tok.encode(text)
        assert actual_ids == expected_ids, f"Encode fuzz mismatch on {text!r}"
        assert tok.decode(actual_ids) == model.decode(expected_ids)
