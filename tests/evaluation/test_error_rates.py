import pytest

from viavsr.evaluation import evaluate_transcript


def test_exact_match_has_zero_error():
    result = evaluate_transcript(
        reference="xin chào các bạn",
        prediction="xin chào các bạn",
    )
    assert result.wer == 0.0
    assert result.cer == 0.0
    assert result.reference_words == 4
    assert result.reference_characters == len("xin chào các bạn")


def test_one_word_deletion():
    result = evaluate_transcript(
        reference="tôi học máy",
        prediction="tôi học",
    )
    assert result.word_deletions == 1
    assert result.wer == pytest.approx(1 / 3)


def test_one_word_substitution():
    result = evaluate_transcript(
        reference="hôm nay trời đẹp",
        prediction="hôm nay trời lạnh",
    )
    assert result.word_substitutions == 1
    assert result.wer == pytest.approx(1 / 4)


def test_one_word_insertion():
    result = evaluate_transcript(
        reference="xin chào bạn",
        prediction="xin chào các bạn",
    )
    assert result.word_insertions == 1
    assert result.wer == pytest.approx(1 / 3)


def test_case_and_whitespace_do_not_create_errors():
    result = evaluate_transcript(
        reference="Hôm Nay Trời Đẹp",
        prediction="hôm   nay trời đẹp",
    )
    assert result.wer == 0.0
    assert result.cer == 0.0


def test_punctuation_does_not_create_errors():
    result = evaluate_transcript(
        reference="Xin chào, các bạn!",
        prediction="xin chào các bạn",
    )
    assert result.wer == 0.0
    assert result.cer == 0.0


def test_diacritic_error_is_counted():
    result = evaluate_transcript(
        reference="tôi đang học",
        prediction="toi đang học",
    )
    assert result.wer > 0
    assert result.cer > 0


def test_empty_reference_and_prediction_have_zero_error():
    result = evaluate_transcript(reference="...", prediction="  ")
    assert result.wer == 0.0
    assert result.cer == 0.0
    assert result.reference_words == 0
    assert result.reference_characters == 0


def test_nonempty_prediction_for_empty_reference_counts_insertions():
    result = evaluate_transcript(reference="", prediction="xin chào")
    assert result.wer == 2.0
    assert result.cer == float(result.char_insertions)
    assert result.word_insertions == 2
    assert result.reference_words == 0
    assert result.reference_characters == 0


def test_result_is_json_serializable():
    result = evaluate_transcript(reference="tôi học", prediction="tôi đọc")
    payload = result.to_dict()
    assert payload["wer"] == pytest.approx(0.5)
    assert payload["word_substitutions"] == 1
