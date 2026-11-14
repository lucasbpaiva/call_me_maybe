import json
from pathlib import Path
from llm_sdk import Small_LLM_Model


CORPUS: list[str] = [
    "",
    "a",
    " ",
    "   \t\r\n  ",
    "Hello world!",
    "Hello , world !",
    "don't",
    "I'll, you've, they're, we'd, he's, I'm",
    "12345.6789",
    "-1.79e308",
    "Café",
    "Cafe\u0301",
    "こんにちは世界",
    "🔥🚀👨‍👩‍👧‍👦",
    "Mixed 日本語 and café 🚀!",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n"
    "<|im_start|>user\nHi<|im_end|>",
    "<think>Let me reason about this.</think>Answer",
    '{"name": "fn_set_thermostat", "parameters": {"room": "cellar", '
    r'"temperature": -2.5, "eco_mode": true}}',
    '", "parameters": {',
]


def main() -> None:
    """Generates and saves golden tokenizer reference cases using the SDK.

    Iterates through a test corpus covering edge cases (whitespace, numbers,
    Unicode normalization forms, CJK, emojis, special tokens, and JSON
    payloads). Encodes and decodes each entry via Small_LLM_Model and writes
    the resulting oracle dataset to tests/golden.json.
    """
    model = Small_LLM_Model()
    golden_records = []
    for text in CORPUS:
        ids: list[int] = model.encode(text).tolist()[0]
        decoded: str = model.decode(ids)
        golden_records.append({
            "text": text,
            "ids": ids,
            "decoded": decoded,
        })
        if any(ord(ch) > 0xFFFF for ch in text):
            for k in range(1, len(ids)):
                prefix_ids = ids[:k]
                golden_records.append({
                    "text": f"<PREFIX {k}/{len(ids)} of {text!r}>",
                    "ids": prefix_ids,
                    "decoded": model.decode(prefix_ids),
                })
    out_path = Path("tests/golden.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(golden_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(golden_records)} golden test cases to {out_path}")


if __name__ == "__main__":
    main()
