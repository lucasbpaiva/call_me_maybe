import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.tokenizer import BPE, Tokenizer


def test_bpe_priority_order() -> None:
    """Verifies lower-index pairs win when adjacent pairs compete.

    Compares two BPE instances with reversed merge rule priorities on a
    three-character string to ensure the pair with the lower rank index
    merges first and consumes the shared middle symbol.
    """
    bpe_ab_first = BPE(merges=[["a", "b"], ["b", "c"]])
    assert bpe_ab_first.merge_word("abc") == ["ab", "c"]
    bpe_bc_first = BPE(merges=[["b", "c"], ["a", "b"]])
    assert bpe_bc_first.merge_word("abc") == ["a", "bc"]


def test_bpe_against_golden_words() -> None:
    """Verifies BPE and vocabulary lookup match golden IDs on single words.

    Loads the production tokenizer and oracle dataset from tests/golden.json,
    converts single-piece test words via ByteMap, applies BPE merges, and
    asserts that the resulting token IDs match the SDK reference output.
    """
    model = Small_LLM_Model()
    tok = Tokenizer.load_from_file(model.get_path_to_tokenizer_file())
    golden = json.loads(Path("tests/golden.json").read_text(encoding="utf-8"))
    single_pieces = {"a", " ", "Hello", "world"}
    for entry in golden:
        text = entry["text"]
        if text in single_pieces:
            mapped = tok.byte_map.bytes_to_str(text.encode("utf-8"))
            symbols = tok.bpe.merge_word(mapped)
            ids = [tok.token_to_id[sym] for sym in symbols]
            assert ids == entry["ids"], f"Mismatch on {text!r}"
