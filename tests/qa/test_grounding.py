from rag_mvp.qa.grounding import validate_grounding


def test_accepts_cited_supported_answer() -> None:
    result = validate_grounding(
        "Employees receive 16 weeks of paid parental leave [1].",
        ["Employees receive 16 weeks of paid parental leave."],
        min_support=0.8,
    )
    assert result.valid is True
    assert result.support_score == 1.0


def test_rejects_missing_and_out_of_range_citations() -> None:
    missing = validate_grounding("Employees receive leave.", ["Employees receive leave."], min_support=0.5)
    invalid = validate_grounding("Employees receive leave [2].", ["Employees receive leave."], min_support=0.5)
    assert missing.reason == "missing_citation"
    assert invalid.reason == "invalid_citation"
