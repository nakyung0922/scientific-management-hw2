"""
Material Collector agent — 단위 + 통합 테스트.

PG2 의 tests/test_common.py 패턴을 따라 mock GeminiClient 로 LLM 없이
end-to-end 동작을 검증한다.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

from common.enums import RoutingStatus
from common.gemini_client import GeminiClient, reset_usage
from common.schemas import MessageEnvelope

from agents.material_collector import (
    AGENT_NAME,
    NEXT_AGENT,
    MaterialCollector,
    MaterialCollectorConfig,
    MaterialCollectorOutput,
)
from agents.material_collector.chunker import make_chunks
from agents.material_collector.concept_id import (
    infer_module_id_from_filename,
    make_concept_id,
    module_id_in_target_chapters,
    slugify,
)
from agents.material_collector.pdf_reader import PageRecord, extract_pages


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _make_text_pdf(path: Path, pages: list[str]) -> None:
    """텍스트 기반 PDF 를 그 자리에서 생성 (OCR 불필요)."""
    doc = fitz.open()
    for content in pages:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((50, 100), content, fontsize=12)
    doc.save(str(path))
    doc.close()


def _planner_mock_responder(system: str, user: str) -> str:
    """section_title 을 그대로 concept_name 으로 돌려주는 mock 응답."""
    import re

    m = re.search(r"Section title \(best guess\): (.+)", user)
    title = m.group(1).strip() if m else "(none)"
    if title and title != "(none)":
        return json.dumps({
            "concept_name": title,
            "summary": f"Mock summary for {title}.",
            "keywords": ["mock", "test"],
            "is_empty": False,
        }, ensure_ascii=False)
    return json.dumps({"concept_name": "", "summary": "", "keywords": [], "is_empty": True})


# ────────────────────────────────────────────────────────────
# Unit tests
# ────────────────────────────────────────────────────────────
def test_slugify_basic():
    """slugify 가 v0.5 규칙(영문/숫자/_만)을 따른다."""
    assert slugify("Taylor's 4 Principles") == "taylors_4_principles"
    assert slugify("KJ Method") == "kj_method"
    # 한국어는 ASCII 가 아니므로 v0.5 규칙상 버려짐
    assert slugify("자동화의 정의") == ""
    # 영어 + 한국어 혼합은 영어만 남음
    assert slugify("KJ Method (브레인스토밍 정리)") == "kj_method"
    print("✓ test_slugify_basic")


def test_concept_id_collision_avoidance():
    """make_concept_id 가 충돌 시 _v2 부여."""
    used: set[str] = set()
    cid1 = make_concept_id("M1_4", "Taylor Principles", used_ids=used)
    used.add(cid1)
    cid2 = make_concept_id("M1_4", "Taylor Principles", used_ids=used)
    assert cid1 == "M1_4_taylor_principles"
    assert cid2 == "M1_4_taylor_principles_v2"
    print("✓ test_concept_id_collision_avoidance")


def test_module_id_inference():
    """파일명에서 module_id 추론."""
    assert infer_module_id_from_filename("M5_1_Automation.pdf") == "M5_1"
    assert infer_module_id_from_filename("M2.1.5_KJ_method.pdf") == "M2_1_5"
    # TARGET_CHAPTERS 안에 있는 자유 이름
    assert infer_module_id_from_filename("manufacturing_overview.pdf") \
        == "manufacturing_overview"
    print("✓ test_module_id_inference")


def test_module_in_target_chapters():
    """시험 범위 검증."""
    assert module_id_in_target_chapters("M1_1") is True
    assert module_id_in_target_chapters("manufacturing_overview") is True
    assert module_id_in_target_chapters("M99_99") is False
    print("✓ test_module_in_target_chapters")


def test_extract_pages_from_text_pdf():
    """텍스트 기반 PDF 추출."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        _make_text_pdf(path, [
            "Taylor's Principles\nFirst content page about scientific management.",
            "Pig Iron Case\nThis is the famous experiment.",
        ])
        pages = extract_pages(path, enable_ocr=False)
        assert len(pages) == 2
        assert all(p.method == "text" for p in pages)
        assert "Taylor" in pages[0].text
        # 첫 줄이 title 로 잡혀야 함
        assert pages[0].title in ("Taylor's Principles", "Taylor")
        print("✓ test_extract_pages_from_text_pdf")
    finally:
        path.unlink()


def test_chunker_windowing():
    """chunker 가 페이지를 묶고 짧은 청크를 머지."""
    pages = [
        PageRecord(1, "Title 1", "A" * 200, "text"),
        PageRecord(2, "Title 2", "B" * 200, "text"),
        PageRecord(3, "Title 3", "C" * 200, "text"),
        PageRecord(4, "Title 4", "D" * 10, "text"),  # 매우 짧음 → 머지됨
    ]
    chunks = make_chunks(pages, pages_per_chunk=2, min_chunk_chars=50)
    assert len(chunks) == 2
    assert chunks[0].page_range == (1, 2)
    # 마지막 청크는 (3,4) 가 됐다가 짧으면 머지되지만, 200+10 이라 충분히 김
    assert chunks[1].page_range == (3, 4)
    print("✓ test_chunker_windowing")


# ────────────────────────────────────────────────────────────
# Integration test — mock LLM 으로 end-to-end
# ────────────────────────────────────────────────────────────
def test_end_to_end_with_mock_llm():
    """텍스트 PDF + mock GeminiClient 로 envelope 까지 전부 만들기."""
    reset_usage()

    with tempfile.NamedTemporaryFile(
        prefix="M1_4_taylorism_", suffix=".pdf", delete=False,
    ) as tmp:
        path = Path(tmp.name)
    try:
        _make_text_pdf(path, [
            "Taylor's Principles\nFirst content page about scientific management.",
            "Pig Iron Case\nFamous experiment about pig iron handling.",
        ])

        client = GeminiClient(mock=True, mock_responder=_planner_mock_responder)
        agent = MaterialCollector(
            client=client,
            config=MaterialCollectorConfig(
                pages_per_chunk=1,           # 페이지마다 청크
                enable_ocr=False,            # 텍스트 PDF 라 OCR 불필요
                min_chunk_chars=10,
            ),
        )
        env = agent.run([str(path)], session_id="test_session_001")

        # envelope 검증
        assert isinstance(env, MessageEnvelope)
        assert env.source_agent == AGENT_NAME
        assert env.target_agent == NEXT_AGENT
        assert env.routing_status == RoutingStatus.FLOW.value
        assert env.session_id == "test_session_001"

        # payload 검증 (Pydantic 으로 다시 검증)
        payload_obj = MaterialCollectorOutput.model_validate(env.payload)
        assert payload_obj.total_pages == 2
        assert payload_obj.total_concepts == 2
        assert payload_obj.extraction_methods == {"text": 2}

        # 첫 concept 의 형식
        c0 = payload_obj.concepts[0]
        # module_id 가 파일명에서 추론됐는지
        assert c0.concept_id.startswith("M1_4_")
        # source_pages 가 "M1_4 p.N" 형식인지
        assert all(sp.startswith("M1_4 p.") for sp in c0.source_pages)
        # mock 이 section_title 을 그대로 concept_name 으로 돌렸으므로
        assert "Taylor" in c0.concept_name

        # round-trip
        serialized = env.model_dump_json()
        env2 = MessageEnvelope.model_validate(json.loads(serialized))
        assert env2.payload["total_concepts"] == 2
        print("✓ test_end_to_end_with_mock_llm")
    finally:
        path.unlink()


def test_outside_target_chapters_warning():
    """시험 범위 밖 PDF 면 warnings 에 기록."""
    reset_usage()

    with tempfile.NamedTemporaryFile(
        prefix="M99_99_unknown_", suffix=".pdf", delete=False,
    ) as tmp:
        path = Path(tmp.name)
    try:
        _make_text_pdf(path, ["Off-topic\nThis is not in the exam."])
        client = GeminiClient(mock=True, mock_responder=_planner_mock_responder)
        agent = MaterialCollector(
            client=client,
            config=MaterialCollectorConfig(
                pages_per_chunk=1, enable_ocr=False, min_chunk_chars=5,
            ),
        )
        env = agent.run([str(path)], session_id="test_warn_001")
        warnings = env.payload["warnings"]
        assert any("module_outside_target_chapters" in w for w in warnings), \
            f"expected warning, got: {warnings}"
        print("✓ test_outside_target_chapters_warning")
    finally:
        path.unlink()


def test_envelope_compatible_with_topic_analyzer_contract():
    """Topic_Analyzer 가 받는 payload 형식 호환성.

    Topic_Analyzer 가 ConceptNode 를 만들 때 필요한 필드들이 ConceptUnit 에
    모두 들어있는지 검증.
    """
    reset_usage()
    with tempfile.NamedTemporaryFile(
        prefix="M1_4_taylor_", suffix=".pdf", delete=False,
    ) as tmp:
        path = Path(tmp.name)
    try:
        _make_text_pdf(path, ["Hard Work\nLong content " * 20])
        client = GeminiClient(mock=True, mock_responder=_planner_mock_responder)
        agent = MaterialCollector(
            client=client,
            config=MaterialCollectorConfig(
                pages_per_chunk=1, enable_ocr=False, min_chunk_chars=5,
            ),
        )
        env = agent.run([str(path)], session_id="contract_check")
        payload = MaterialCollectorOutput.model_validate(env.payload)
        for unit in payload.concepts:
            # Topic_Analyzer.ConceptNode 와의 연결고리
            assert unit.concept_id            # → ConceptNode.concept_id
            assert unit.concept_name          # → ConceptNode.concept_name
            assert unit.source_pages          # → ConceptNode.source_pages
            assert unit.summary               # 임베딩 입력
            assert isinstance(unit.keywords, list)
        print("✓ test_envelope_compatible_with_topic_analyzer_contract")
    finally:
        path.unlink()


# ────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_slugify_basic()
    test_concept_id_collision_avoidance()
    test_module_id_inference()
    test_module_in_target_chapters()
    test_extract_pages_from_text_pdf()
    test_chunker_windowing()
    test_end_to_end_with_mock_llm()
    test_outside_target_chapters_warning()
    test_envelope_compatible_with_topic_analyzer_contract()
    print("\n✓ All Material Collector tests passed")
