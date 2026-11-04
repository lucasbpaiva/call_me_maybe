import re


class Node:
    """Represents a single character node within the PrefixTree."""
    def __init__(self) -> None:
        """Initializes a new Trie node with empty children."""
        self.children: dict[str, 'Node'] = {}
        self.is_leaf: bool = False


class PrefixTree:
    """A Trie DS for efficient prefix validation and character lookups."""
    def __init__(self) -> None:
        """Initializes the PrefixTree with an empty root node."""
        self.root = Node()

    def insert(self, key: str) -> None:
        """Inserts a complete string key into the prefix tree.

        Args:
            key: The string to insert into the tree.
        """
        curr = self.root
        for c in key:
            if c not in curr.children:
                curr.children[c] = Node()
            curr = curr.children[c]
        curr.is_leaf = True

    def search(self, key: str) -> bool:
        """Searches for an exact, complete string match in the prefix tree.

        Args:
            key: The string to search for.

        Returns:
            True if the complete string exists in the tree as a leaf,
            False otherwise.
        """
        curr = self.root
        for c in key:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return curr.is_leaf

    def is_prefix(self, prefix: str) -> bool:
        """Determines if given prefix is a valid prefix of any inserted key.

        Args:
            prefix: The prefix string to evaluate.

        Returns:
            True if the prefix forms a valid path in the tree,
            False otherwise.
        """
        curr = self.root
        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return True

    def get_next_characters(self, prefix: str) -> list[str]:
        r"""Retrieves all valid continuation characters for a given prefix.

        If the prefix itself constitutes a complete valid key (a leaf node),
        a null character '\0' is included in the returned list to signal a
        valid stopping point.

        Args:
            prefix: The prefix string to evaluate.

        Returns:
            A list of single-character strings representing valid
            continuations. Returns an empty list if the prefix is invalid.
        """
        next_chars = []
        curr = self.root
        if self.search(prefix):
            next_chars.append('\0')
        for c in prefix:
            if c not in curr.children:
                return []
            curr = curr.children[c]
        for char in curr.children:
            next_chars.append(char)
        return next_chars


class NumberValidator:
    """A state machine for validating partial and complete JSON numbers.

    This class uses regular expressions to determine whether a given string
    is a valid mid-generation (partial) number or a fully parseable complete
    number according to strict JSON grammar.

    Attributes:
        partial_pattern: Regex pattern matching valid incomplete numbers.
        complete_pattern: Regex pattern matching strictly valid numbers.
    """
    def __init__(self) -> None:
        """Initializes the NumberValidator with compiled regex patterns."""
        # Matches: "-", "-0", "4", "4.", "-4.5", but rejects "04"
        self.partial_pattern = re.compile(r"^-?(?:0|[1-9]\d*)?(?:\.\d*)?$")
        # Matches strictly valid JSON numbers: "4", "-0", "4.5"
        self.complete_pattern = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")

    def is_partial(self, text: str) -> bool:
        """Determines if the text can continue to become a valid number.

        Args:
            text: The string representing the currently generated sequence.

        Returns:
            True if the text represents a valid partial number or is empty,
            False otherwise.
        """
        if text == "":
            return True
        return bool(self.partial_pattern.match(text))

    def is_complete(self, text: str) -> bool:
        """Determines if the text is a complete, parseable number.

        Args:
            text: The string to validate.

        Returns:
            True if the text exactly matches the complete JSON number syntax,
            False otherwise.
        """
        return bool(self.complete_pattern.match(text))
