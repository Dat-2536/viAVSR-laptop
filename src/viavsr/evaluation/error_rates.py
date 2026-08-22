import jiwer

from .schemas import ErrorRateResult
from .text_normalization import normalize_vietnamese_text


def _error_rate(
    substitutions: int,
    deletions: int,
    insertions: int,
    reference_length: int,
) -> float:
    """Calculate an error rate, including a defined empty-reference policy."""
    errors = substitutions + deletions + insertions
    if reference_length == 0:
        return float(insertions)
    return errors / reference_length


def evaluate_transcript(reference: str, prediction: str) -> ErrorRateResult:
    """Compute normalized Vietnamese WER/CER for one utterance.

    Vietnamese diacritics are preserved by normalization. For an empty
    normalized reference, an empty prediction has zero error; otherwise the
    rate is the number of inserted words or characters, matching JiWER 4.
    """
    normalized_reference = normalize_vietnamese_text(reference)
    normalized_prediction = normalize_vietnamese_text(prediction)
    word_result = jiwer.process_words(normalized_reference, normalized_prediction)
    char_result = jiwer.process_characters(normalized_reference, normalized_prediction)

    reference_words = (
        word_result.hits + word_result.substitutions + word_result.deletions
    )
    reference_characters = (
        char_result.hits + char_result.substitutions + char_result.deletions
    )

    return ErrorRateResult(
        wer=_error_rate(
            word_result.substitutions,
            word_result.deletions,
            word_result.insertions,
            reference_words,
        ),
        cer=_error_rate(
            char_result.substitutions,
            char_result.deletions,
            char_result.insertions,
            reference_characters,
        ),
        word_substitutions=word_result.substitutions,
        word_deletions=word_result.deletions,
        word_insertions=word_result.insertions,
        char_substitutions=char_result.substitutions,
        char_deletions=char_result.deletions,
        char_insertions=char_result.insertions,
        reference_words=reference_words,
        reference_characters=reference_characters,
    )
