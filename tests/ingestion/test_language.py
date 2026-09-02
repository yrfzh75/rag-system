from rag_mvp.ingestion.language import detect_language


def test_detects_english_chinese_and_mixed_text() -> None:
    assert detect_language("Employees receive parental leave benefits.") == "en"
    assert detect_language("员工可以享受育儿假和相关福利。") == "zh"
    assert detect_language("员工 handbook 说明 parental leave 政策。") == "mixed"
