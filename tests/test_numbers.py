from src.constraints import NumberValidator


def test_number_state_machine() -> None:
    """Tests the NumberValidator state machine for JSON number constraints.

    Verifies that the validator correctly identifies fully parseable
    JSON numbers, accepts valid partial sequences during mid-generation,
    and immediately rejects invalid syntax such as leading zeros or
    malformed decimal structures.
    """
    validator = NumberValidator()

    assert validator.is_complete("40") is True
    assert validator.is_complete("-0") is True
    assert validator.is_complete("3.14") is True
    assert validator.is_complete("-9999.99") is True
    assert validator.is_complete("0") is True

    assert validator.is_partial("-") is True
    assert validator.is_complete("-") is False

    assert validator.is_partial("40.") is True
    assert validator.is_complete("40.") is False

    assert validator.is_partial("-0.") is True
    assert validator.is_complete("-0.") is False

    assert validator.is_partial("04") is False  # JSON forbids leading zeros
    assert validator.is_partial(".-5") is False
    assert validator.is_partial("3.14.15") is False
