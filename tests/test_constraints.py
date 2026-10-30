import pytest
from src.constraints import PrefixTree


@pytest.fixture
def function_names() -> list[str]:
    return [
        "fn_add",
        "fn_add_numbers",
        "fn_greet",
        "fn_get_square_root"
    ]


@pytest.fixture
def trie(function_names: list[str]) -> PrefixTree:
    tree = PrefixTree()
    for name in function_names:
        tree.insert(name)
    return tree


def test_exact_leaf_matches(trie: PrefixTree) -> None:
    """Test that complete, valid function names are recognized as leaves."""
    assert trie.search("fn_add") is True
    assert trie.search("fn_add_numbers") is True
    assert trie.search("fn_greet") is True
    # Incomplete paths are not leaves
    assert trie.search("fn_g") is False
    assert trie.search("fn_add_") is False
    assert trie.search("invalid") is False


def test_valid_prefixes(trie: PrefixTree) -> None:
    """Test that partial strings correctly evaluate as valid paths."""
    assert trie.is_prefix("") is True
    assert trie.is_prefix("f") is True
    assert trie.is_prefix("fn_") is True
    assert trie.is_prefix("fn_a") is True
    assert trie.is_prefix("fn_add") is True
    assert trie.is_prefix("fn_add_num") is True


def test_invalid_prefixes(trie: PrefixTree) -> None:
    """Test that off-path strings are instantly rejected."""
    assert trie.is_prefix("fn_x") is False
    assert trie.is_prefix("add") is False
    assert trie.is_prefix("fn_greet_person") is False


def test_get_next_characters_branching(trie: PrefixTree) -> None:
    """Test extraction of allowed next characters at a branching node."""
    # After "fn_g", the next valid chars are 'r' (greet)
    # or 'e' (get_square_root)
    next_chars = trie.get_next_characters("fn_g")
    assert set(next_chars) == {"r", "e"}


def test_get_next_characters_overlapping_names(trie: PrefixTree) -> None:
    """Test a node can simultaneously be a valid leaf and have children."""
    # "fn_add" is a complete function, but also the prefix for "fn_add_numbers"
    # We expect an end-of-string token ('\0') AND a next valid character ('_')
    next_chars = trie.get_next_characters("fn_add")
    assert "_" in next_chars
    assert "\0" in next_chars


def test_get_next_characters_invalid_prefix(trie: PrefixTree) -> None:
    """Test that an invalid prefix returns no valid continuations."""
    assert trie.get_next_characters("fn_unknown") == []
