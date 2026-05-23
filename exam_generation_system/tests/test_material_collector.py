"""
Unit tests for Material_Collector.

Mock 모드. 실제 PDF 파싱은 fixtures의 작은 stub PDF로 검증.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.material_collector import (
    MaterialCollector,
    SummaryRawOutput,
    clean_page_text,
    infer_module_id,
    parse_pdf_to_raw_document,
)
from common.gemini_client import GeminiClient, _tracker, reset_usage
from common.schemas import ConceptSummary, PageText, RawDocument

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ────────────────────────────────────────────────────────────
# 1. infer_module_id (파일명 휴리스틱)
# ────────────────────────────────────────────────────────────
def test_infer_module_id_standard():
    assert infer_module_id("M1_1_What_is_Work.pdf") == "M1_1"
    assert infer_module_id("M1_4_Scientific_Management.pdf") == "M1_4"
    print("✓ test_infer_module_id_standard")


def test_infer_module_id_deep():
    assert infer_module_id("M2_1_2_KJ_Method.pdf") == "M2_1_2"
    assert infer_module_id("M3_1_1_Therbligs.pdf") == "M3_1_1"
    print("✓ test_infer_module_id_deep")


def test_infer_module_id_no_prefix():
    """M 접두사 없으면 파일명 stem을 정제해서 사용."""
    assert infer_module_id("manufacturing_overview.pdf") == "manufacturing_overview"
    print("✓ test_infer_module_id_no_prefix")


def test_infer_module_id_dots_and_spaces():
    """점·공백은 _로 변환."""
    assert infer_module_id("some random file.pdf") == "some_random_file"
    print("✓ test_infer_module_id_dots_and_spaces")


# ────────────────────────────────────────────────────────────
# 2. clean_page_text (정제 로직)
# ────────────────────────────────────────────────────────────
def test_clean_page_text_removes_page_numbers():
    raw = "Real content here\n3\nMore content"
    cleaned = clean_page_text(raw)
    assert "Real content here" in cleaned
    assert "More content" in cleaned
    assert "\n3\n" not in cleaned
    print("✓ test_clean_page_text_removes_page_numbers")


def test_clean_page_text_collapses_whitespace():
    raw = "Line 1\n\n\n\nLine 2\n  \n  Line 3"
    cleaned = clean_page_text(raw)
    lines = cleaned.split("\n")
    assert len(lines) == 3  # 3개 의미 있는 라인
    print("✓ test_clean_page_text_collapses_whitespace")


def test_clean_page_text_removes_short_fragments():
    raw = "Valid content\na\nb\nMore valid content"
    cleaned = clean_page_text(raw)
    assert "Valid content" in cleaned
    assert "More valid content" in cleaned
    assert "\na\n" not in cleaned
    print("✓ test_clean_page_text_removes_short_fragments")


# ────────────────────────────────────────────────────────────
# 3. PDF 파싱 (fixture PDF가 있을 때만 동작)
# ────────────────────────────────────────────────────────────
def test_parse_pdf_missing_file():
    """존재하지 않는 PDF → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_pdf_to_raw_document(Path("/nonexistent/file.pdf"))
    print("✓ test_parse_pdf_missing_file")


def test_parse_pdf_real_file(tmp_path):
    """실제 작은 PDF 생성 후 파싱 (reportlab 필요).

    reportlab이 없으면 skip — fixtures/ 디렉토리에 미리 PDF를 두는 대안도 가능.
    """
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    pdf_path = tmp_path / "M1_1_Test_Module.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Page 1 content about scientific management")
    c.drawString(100, 730, "Taylor's four principles")
    c.showPage()
    c.drawString(100, 750, "Page 2 content")
    c.drawString(100, 730, "Pig Iron Case study")
    c.showPage()
    c.save()

    raw_doc = parse_pdf_to_raw_document(pdf_path)
    assert raw_doc.module_id == "M1_1"
    assert raw_doc.source_filename == "M1_1_Test_Module.pdf"
    assert len(raw_doc.pages) == 2
    assert "Taylor" in raw_doc.pages[0].text or "scientific" in raw_doc.pages[0].text.lower()
    print("✓ test_parse_pdf_real_file")


# ────────────────────────────────────────────────────────────
# 4. MaterialCollector.collect() — mock LLM
# ────────────────────────────────────────────────────────────
def _make_responder(title: str, summary: str, phrases: list[str]):
    """주어진 값으로 응답하는 mock responder."""
    def responder(system: str, user: str) -> str:
        return json.dumps({
            "title": title,
            "summary_text": summary,
            "key_phrases": phrases,
        })
    return responder


def test_collect_single_pdf(tmp_path):
    """1 PDF → 1 ConceptSummary."""
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    pdf_path = tmp_path / "M1_4_Taylor.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Taylor's scientific management principles")
    c.showPage()
    c.save()

    responder = _make_responder(
        title="Taylor's Scientific Management",
        summary="Taylor의 4원칙을 다루는 모듈. 과학화, 선발/훈련, 협력, 분업.",
        phrases=["Taylor의 4원칙", "과학화", "선발 훈련", "노사 협력", "분업"],
    )
    collector = MaterialCollector(
        GeminiClient(mock=True, mock_responder=responder)
    )
    summaries = collector.collect([pdf_path])

    assert len(summaries) == 1
    s = summaries[0]
    assert isinstance(s, ConceptSummary)
    assert s.module_id == "M1_4"
    assert s.title == "Taylor's Scientific Management"
    assert len(s.key_phrases) == 5
    assert s.page_count == 1
    print("✓ test_collect_single_pdf")


def test_collect_missing_pdf_fallback():
    """존재하지 않는 PDF → fallback ConceptSummary (warnings 포함)."""
    collector = MaterialCollector(
        GeminiClient(mock=True, mock_responder=_make_responder("x", "x", ["a"]*5))
    )
    summaries = collector.collect([Path("/nonexistent/M1_1_fake.pdf")])

    assert len(summaries) == 1
    s = summaries[0]
    assert s.module_id == "M1_1"
    assert "parse_failure" in s.warnings[0]
    assert s.key_phrases == []
    print("✓ test_collect_missing_pdf_fallback")


def test_collect_uses_light_model(tmp_path):
    """usage_tracker에서 light model 호출 확인 (Material_Collector → light)."""
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    reset_usage()

    pdf_path = tmp_path / "M1_2_test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "test content")
    c.showPage()
    c.save()

    responder = _make_responder("T", "S", ["p1", "p2", "p3", "p4", "p5"])
    collector = MaterialCollector(GeminiClient(mock=True, mock_responder=responder))
    collector.collect([pdf_path])

    records = [r for r in _tracker.records if r.agent_name == "Material_Collector"]
    assert len(records) == 1
    # config/defaults.py의 agent_model_assignment에 Material_Collector가 없으므로 default light
    from config.defaults import LLM_CONFIG
    light = LLM_CONFIG["models"]["light"]
    assert light in records[0].model
    print("✓ test_collect_uses_light_model")


def test_collect_too_few_phrases_warning(tmp_path):
    """key_phrases 5개 미만 → warning 발생, 흐름 중단 안 함."""
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    pdf_path = tmp_path / "M1_3_test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "content")
    c.showPage()
    c.save()

    responder = _make_responder("T", "S", ["only one phrase"])  # 1개만
    collector = MaterialCollector(GeminiClient(mock=True, mock_responder=responder))
    summaries = collector.collect([pdf_path])

    assert len(summaries) == 1
    assert any("부족" in w for w in summaries[0].warnings)
    print("✓ test_collect_too_few_phrases_warning")


def test_collect_too_many_phrases_truncated(tmp_path):
    """key_phrases 15개 초과 → 상위 15개로 자름 + warning."""
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    pdf_path = tmp_path / "M1_5_test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "content")
    c.showPage()
    c.save()

    many_phrases = [f"phrase_{i}" for i in range(25)]
    responder = _make_responder("T", "S", many_phrases)
    collector = MaterialCollector(GeminiClient(mock=True, mock_responder=responder))
    summaries = collector.collect([pdf_path])

    assert len(summaries[0].key_phrases) == 15
    assert any("너무 많음" in w for w in summaries[0].warnings)
    print("✓ test_collect_too_many_phrases_truncated")


def test_collect_llm_validation_error_fallback(tmp_path):
    """LLM이 깨진 JSON → fallback summary (전체 흐름 멈추지 않음)."""
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    pdf_path = tmp_path / "M2_1_1_test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "DASSI five steps")
    c.showPage()
    c.save()

    def broken_responder(system: str, user: str) -> str:
        # 필수 필드 title 누락
        return json.dumps({"summary_text": "x", "key_phrases": ["a"]*5})

    collector = MaterialCollector(GeminiClient(mock=True, mock_responder=broken_responder))
    summaries = collector.collect([pdf_path])

    assert len(summaries) == 1
    assert any("llm_failure" in w or "fallback_used" in w for w in summaries[0].warnings)
    print("✓ test_collect_llm_validation_error_fallback")


def test_collect_multiple_pdfs(tmp_path):
    """여러 PDF → 입력 순서대로 ConceptSummary 리스트."""
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    paths = []
    for mid in ["M1_1", "M1_2", "M1_3"]:
        p = tmp_path / f"{mid}_test.pdf"
        c = canvas.Canvas(str(p))
        c.drawString(100, 750, f"content of {mid}")
        c.showPage()
        c.save()
        paths.append(p)

    responder = _make_responder("T", "S", ["a", "b", "c", "d", "e"])
    collector = MaterialCollector(GeminiClient(mock=True, mock_responder=responder))
    summaries = collector.collect(paths)

    assert len(summaries) == 3
    assert [s.module_id for s in summaries] == ["M1_1", "M1_2", "M1_3"]
    print("✓ test_collect_multiple_pdfs")


# ────────────────────────────────────────────────────────────
# 5. Schema 호환성
# ────────────────────────────────────────────────────────────
def test_concept_summary_serialization():
    """ConceptSummary가 JSON 직렬화·역직렬화 가능."""
    s = ConceptSummary(
        module_id="M1_1",
        source_filename="x.pdf",
        title="Test",
        summary_text="Some summary",
        key_phrases=["a", "b"],
        page_count=3,
    )
    serialized = s.model_dump_json()
    rehydrated = ConceptSummary.model_validate_json(serialized)
    assert rehydrated.module_id == s.module_id
    print("✓ test_concept_summary_serialization")


# ────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_infer_module_id_standard()
    test_infer_module_id_deep()
    test_infer_module_id_no_prefix()
    test_infer_module_id_dots_and_spaces()
    test_clean_page_text_removes_page_numbers()
    test_clean_page_text_collapses_whitespace()
    test_clean_page_text_removes_short_fragments()
    test_parse_pdf_missing_file()
    test_concept_summary_serialization()
    print("\n✓ All Material_Collector tests passed (non-PDF subset)")
