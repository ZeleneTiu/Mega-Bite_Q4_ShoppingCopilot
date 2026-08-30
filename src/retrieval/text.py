"""Shared text normalisation for the retrieval layer.

Every index in this package tokenises through these helpers, so the
lexical index, the category index and the query parser can never drift
apart on what counts as a token.
"""
from __future__ import annotations

import re

# Letters and digits only. Deliberately splits on punctuation so that
# catalog strings like "Material:alloy" become ["material", "alloy"],
# which is how the simulated customer's phrasing arrives too.
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Kept close to the starter agent's list so scores stay comparable to the
# published baseline. These carry no retrieval signal in product text.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "im", "need", "want", "key", "requirement", "still", "exploring",
})


def flatten(value: object) -> str:
    """Collapse a catalog field into one string.

    Catalog fields are inconsistently typed: ``details`` is a dict,
    ``features`` and ``categories`` are lists, ``title`` and ``store`` are
    plain strings. This mirrors the evaluator's own flattening so the text
    we index matches the text the customer simulator quotes from.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace, for phrase containment checks."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenise(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lowercase token list, single characters and stopwords dropped."""
    tokens = TOKEN_RE.findall(text.lower())
    if keep_stopwords:
        return [token for token in tokens if len(token) > 1]
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]
