from .schemas import ErrorRateResult
from .text_normalization import normalize_vietnamese_text


def evaluate_transcript(reference: str, prediction: str) -> ErrorRateResult:
    """Compute Vietnamese WER/CER for one utterance.

    VIASVR-6 skeleton.

    TODO:
    1. Normalize reference and prediction.
    2. Compute word-level substitutions/deletions/insertions.
    3. Compute character-level substitutions/deletions/insertions.
    4. Compute WER and CER.
    5. Define behavior for an empty normalized reference.
    6. Return ErrorRateResult.

    Recommended library: `jiwer`.
    """
    _reference = normalize_vietnamese_text(reference)
    _prediction = normalize_vietnamese_text(prediction)

    raise NotImplementedError(
        "VIASVR-6 TODO: implement WER/CER and edit-operation counts."
    )
