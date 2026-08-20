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

    def to_dict(self) -> dict:
        """Convert the result to a JSON-serializable dictionary."""
        return asdict(self)



# __repr__ for DEBUG
    def __repr__(self) -> str:
        return (
            f"ErrorRateResult("
            f"WER={self.wer:.2%}, "
            f"CER={self.cer:.2%}, "
            f"words={self.reference_words}, "
            f"chars={self.reference_characters}, "
            f"word_errors=(S:{self.word_substitutions}, "
            f"D:{self.word_deletions}, "
            f"I:{self.word_insertions}), "
            f"char_errors=(S:{self.char_substitutions}, "
            f"D:{self.char_deletions}, "
            f"I:{self.char_insertions})"
            f")"
        )
