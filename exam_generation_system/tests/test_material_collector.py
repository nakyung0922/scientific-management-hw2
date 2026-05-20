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
    is_valid_id_format,
    label_module_from_filename,
    make_concept_id,
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


def test_label_module_from_filename():
    """파일명 → module_id 라벨링.

    Material Collector 는 시험 범위에 대해 가정하지 않고, 어떤 파일이 들어와도
    안정적인 라벨을 만든다.
    """
    # 1) M{숫자}.{숫자} 모듈 코드는 점 → 언더스코어
    assert label_module_from_filename("M1.4.pdf") == "M1_4"
    assert label_module_from_filename("M2.1.2_1.pdf") == "M2_1_2_1"
    assert label_module_from_filename("M5_1_Automation.pdf") == "M5_1_automation"
    # 2) CamelCase 파일명은 분해해서 snake_case
    assert label_module_from_filename("IntroductionOverviewManufacturing.pdf") \
        == "introduction_overview_manufacturing"
    # 3) 공백/대문자가 섞인 일반 파일명
    assert label_module_from_filename("KJ Method.pdf") == "kj_method"
    print("✓ test_label_module_from_filename")


def test_label_module_collision_avoidance():
    """같은 라벨이 두 번 나오면 _2, _3 부여."""
    used: set[str] = set()
    a = label_module_from_filename("M1.4.pdf", used_labels=used)
    used.add(a)
    b = label_module_from_filename("M1.4.pdf", used_labels=used)
    used.add(b)
    c = label_module_from_filename("M1.4.pdf", used_labels=used)
    assert a == "M1_4"
    assert b == "M1_4_2"
    assert c == "M1_4_3"
    print("✓ test_label_module_collision_avoidance")


def test_label_korean_filename_fallback():
    """한글만 있는 파일명은 ASCII 가 없어서 material_<hash> 형태로 fallback."""
    label = label_module_from_filename("자동화_개론.pdf")
    assert label.startswith("material_")
    assert is_valid_id_format(label)
    # 같은 파일명은 같은 라벨이 나와야 함 (안정적 해시)
    label2 = label_module_from_filename("자동화_개론.pdf")
    assert label == label2
    print("✓ test_label_korean_filename_fallback")


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
        # module_id 가 파일명에서 라벨링되어 concept_id 에 들어감
        assert c0.concept_id.startswith("M1_4"), f"got {c0.concept_id!r}"
        # source_pages 는 'M1_4... p.N' 형식 (라벨이 길어졌어도 동일 prefix)
        assert all(sp.startswith("M1_4") and " p." in sp for sp in c0.source_pages), \
            f"got {c0.source_pages!r}"
        # mock 이 section_title 을 그대로 concept_name 으로 돌렸으므로
        assert "Taylor" in c0.concept_name
        # 새로 추가된 file_to_module_id 매핑 확인 — tempfile 의 random suffix 때문에
        # 정확히 'M1_4' 가 아니라 'M1_4_taylorism_xxxxx' 형태가 될 수 있음.
        assigned = payload_obj.file_to_module_id[path.name]
        assert assigned.startswith("M1_4"), f"got {assigned!r}"

        # round-trip
        serialized = env.model_dump_json()
        env2 = MessageEnvelope.model_validate(json.loads(serialized))
        assert env2.payload["total_concepts"] == 2
        print("✓ test_end_to_end_with_mock_llm")
    finally:
        path.unlink()


def test_arbitrary_filename_does_not_warn():
    """시험 범위와 무관하게 어떤 파일이 들어와도 정상 처리.

    이전 버전은 'TARGET_CHAPTERS 밖' warning 을 띄웠지만, Material Collector 는
    그 가정을 버렸다. 파일이 무엇이든 라벨링하고 처리한다.
    """
    reset_usage()

    with tempfile.NamedTemporaryFile(
        prefix="SomeRandomName_", suffix=".pdf", delete=False,
    ) as tmp:
        path = Path(tmp.name)
    try:
        _make_text_pdf(path, [
            "Random Topic\nSome content about a random topic.",
        ])
        client = GeminiClient(mock=True, mock_responder=_planner_mock_responder)
        agent = MaterialCollector(
            client=client,
            config=MaterialCollectorConfig(
                pages_per_chunk=1, enable_ocr=False, min_chunk_chars=5,
            ),
        )
        env = agent.run([str(path)], session_id="test_arb_001")
        payload = MaterialCollectorOutput.model_validate(env.payload)

        # warning 은 옛 "module_outside_target_chapters" 가 아니어야 함
        for w in payload.warnings:
            assert "module_outside_target_chapters" not in w, \
                f"deprecated warning surfaced: {w}"
        # concept 은 정상 추출돼야 함
        assert payload.total_concepts >= 1
        # file_to_module_id 매핑이 있어야 함
        assert path.name in payload.file_to_module_id
        print("✓ test_arbitrary_filename_does_not_warn")
    finally:
        path.unlink()


def test_korean_filename_auto_labeled():
    """한글만 들어간 파일명은 material_<hash> 로 자동 라벨링 + 정보성 warning."""
    reset_usage()

    # 한글 파일명을 직접 만들기
    tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / "자동화_개론.pdf"
    try:
        _make_text_pdf(path, [
            "Automation\nThis is content about automation.",
        ])
        client = GeminiClient(mock=True, mock_responder=_planner_mock_responder)
        agent = MaterialCollector(
            client=client,
            config=MaterialCollectorConfig(
                pages_per_chunk=1, enable_ocr=False, min_chunk_chars=5,
            ),
        )
        env = agent.run([str(path)], session_id="test_kor_001")
        payload = MaterialCollectorOutput.model_validate(env.payload)

        assigned = payload.file_to_module_id["자동화_개론.pdf"]
        assert assigned.startswith("material_")
        # auto_labeled warning 이 떠야 함 (에러 아닌 info)
        assert any("auto_labeled" in w for w in payload.warnings), \
            f"expected auto_labeled warning, got: {payload.warnings}"
        print("✓ test_korean_filename_auto_labeled")
    finally:
        if path.exists():
            path.unlink()
        tmp_dir.rmdir()


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
    test_label_module_from_filename()
    test_label_module_collision_avoidance()
    test_label_korean_filename_fallback()
    test_extract_pages_from_text_pdf()
    test_chunker_windowing()
    test_end_to_end_with_mock_llm()
    test_arbitrary_filename_does_not_warn()
    test_korean_filename_auto_labeled()
    test_envelope_compatible_with_topic_analyzer_contract()
    print("\n✓ All Material Collector tests passed")
