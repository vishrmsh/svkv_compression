import pytest
from svkv.tasks import normalize, score_answer


def test_needle_score_does_not_accept_digits_embedded_in_longer_number():
    assert score_answer("niah_single", "12345678", ["1234567"]) == 0
    assert score_answer("niah_single", "The code is 1234567.", ["1234567"]) == 1
    assert score_answer("niah_multi", "1234567", ["1234567", "7654321"]) == .5


def test_qa_uses_best_reference_token_f1_not_substring_match():
    assert normalize("The Red, a fox!") == "red fox"
    assert score_answer("hotpotqa", "The Red Fox", ["blue fox", "red fox"]) == 1
    assert score_answer("qasper", "red", ["red fox"]) == pytest.approx(2/3)
    assert score_answer("qasper", "unanswerable", ["unanswerable"]) == 1
    assert score_answer("qasper", "", ["answer"]) == 0


def test_retrieval_multiple_guesses_are_penalized_as_in_longbench():
    assert score_answer("passage_retrieval_en", "Paragraph 12", ["Paragraph 12"]) == 1
    assert score_answer("passage_retrieval_en", "Paragraph 12 or 13", ["Paragraph 12"]) == .5
    assert score_answer("passage_retrieval_en", "Paragraph 1", ["Paragraph 12"]) == 0
    assert score_answer("passage_retrieval_en", "Unknown", ["Paragraph 12"]) == 0
