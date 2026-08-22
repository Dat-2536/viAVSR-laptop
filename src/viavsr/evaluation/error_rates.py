from .schemas import ErrorRateResult
from .text_normalization import normalize_vietnamese_text

import jiwer

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


    wer = jiwer.wer(_reference, _prediction)
    cer = jiwer.cer(_reference, _prediction)

    word_result =  jiwer.process_words(_reference, _prediction)
    char_result =  jiwer.process_characters(_reference, _prediction)



    return ErrorRateResult(
        wer=word_result.wer,
        cer=char_result.cer,

        word_substitutions=word_result.substitutions,
        word_deletions=word_result.deletions,
        word_insertions=word_result.insertions,

        char_substitutions=char_result.substitutions,
        char_deletions=char_result.deletions,
        char_insertions=char_result.insertions,

        reference_words=(
            word_result.hits
            + word_result.substitutions
            + word_result.deletions
        ),
        reference_characters=(
            char_result.hits
            + char_result.substitutions
            + char_result.deletions
        ),
    )


    raise NotImplementedError(
        "VIASVR-6 TODO: implement WER/CER and edit-operation counts."
    )
