from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ErrorRateResult:
    """Structured result returned by transcript evaluation."""

    wer: float
    cer: float

    word_substitutions: int
    word_deletions: int
    word_insertions: int

    char_substitutions: int
    char_deletions: int
    char_insertions: int

    reference_words: int
    reference_characters: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert the result to a JSON-serializable dictionary."""
        return asdict(self)
