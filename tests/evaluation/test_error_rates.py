import pytest

from viasvr.evaluation import evaluate_transcript


@pytest.mark.skip(reason="VIASVR-6 skeleton: implement WER/CER first.")
def test_exact_match_has_zero_error():
    result = evaluate_transcript(
        reference="xin chào các bạn",
        prediction="xin chào các bạn",
    )
    assert result.wer == 0.0
    assert result.cer == 0.0


@pytest.mark.skip(reason="VIASVR-6 skeleton: implement WER/CER first.")
def test_one_word_deletion():
    result = evaluate_transcript(
        reference="tôi học máy",
        prediction="tôi học",
    )
    assert result.word_deletions == 1
    assert result.wer == pytest.approx(1 / 3)


@pytest.mark.skip(reason="VIASVR-6 skeleton: implement WER/CER first.")
def test_one_word_substitution():
    result = evaluate_transcript(
        reference="hôm nay trời đẹp",
        prediction="hôm nay trời lạnh",
    )
    assert result.word_substitutions == 1
    assert result.wer == pytest.approx(1 / 4)


@pytest.mark.skip(reason="VIASVR-6 skeleton: implement WER/CER first.")
def test_case_and_whitespace_do_not_create_errors():
    result = evaluate_transcript(
        reference="Hôm Nay Trời Đẹp",
        prediction="hôm   nay trời đẹp",
    )
    assert result.wer == 0.0


@pytest.mark.skip(reason="VIASVR-6 skeleton: implement WER/CER first.")
def test_diacritic_error_is_counted():
    result = evaluate_transcript(
        reference="tôi đang học",
        prediction="toi đang học",
    )
    assert result.wer > 0
    assert result.cer > 0
