"""
Unit tests for Req_Parser.

Mock LLM 모드. 모든 입력 (dict | str | Path | None) + 모든 fallback 경로를 검증.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.req_parser import (
    ReqParser,
    _input_to_text,
    normalize_target,
    normalize_target_list,
)
from common.enums import EvaluationFocus
from common.gemini_client import GeminiClient, _tracker, reset_usage
from common.schemas import (
    DifficultyDistributionGroup,
    QuestionTypeDistribution,
    ReqVector,
)
from config.defaults import (
    DIFFICULTY_DISTRIBUTION_GROUP,
    EXAM_BASICS,
    TARGET_CHAPTERS,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ────────────────────────────────────────────────────────────
# Mock LLM responder factory
# ────────────────────────────────────────────────────────────
def _make_responder(payload: dict):
    """주어진 payload를 JSON으로 반환하는 mock responder."""
    def responder(system: str, user: str) -> str:
        return json.dumps(payload, ensure_ascii=False)
    return responder


def _all_null_responder():
    """모든 필드 null인 응답 (defaults 적용 흐름 테스트용)."""
    return _make_responder({
        "exam_title": None,
        "exam_type": None,
        "target_chapters_or_concepts": None,
        "total_points": None,
        "duration_minutes": None,
        "evaluation_focus": None,
        "difficulty_distribution_group": None,
        "question_type_distribution": None,
        "allowed_formats": None,
        "special_instructions": None,
        "_warnings": [],
    })


def _full_responder():
    """완전한 응답 (defaults 안 쓰는 케이스)."""
    return _make_responder({
        "exam_title": "Test Exam",
        "exam_type": "midterm",
        "target_chapters_or_concepts": ["M1_1", "M1_2"],
        "total_points": 100,
        "duration_minutes": 75,
        "evaluation_focus": "balanced",
        "difficulty_distribution_group": {"easy": 2, "medium": 6, "hard": 2},
        "question_type_distribution": {
            "MCQ_single": 0, "short_answer": 2, "long_answer": 5, "case_analysis": 3,
        },
        "allowed_formats": ["docx", "pdf"],
        "special_instructions": None,
        "_warnings": [],
    })


# ────────────────────────────────────────────────────────────
# 1. normalize_target — 정규화 유닛 테스트
# ────────────────────────────────────────────────────────────
def test_normalize_target_module_id():
    assert normalize_target("M1_1") == "M1_1"
    assert normalize_target("m1.1") == "M1_1"
    assert normalize_target("M1-1") == "M1_1"
    assert normalize_target("M2_1_2") == "M2_1_2"
    print("✓ test_normalize_target_module_id")


def test_normalize_target_concept_id():
    assert normalize_target("M2_1_2_kj_method") == "M2_1_2_kj_method"
    assert normalize_target("M2_1_2_KJ_Method") == "M2_1_2_kj_method"
    print("✓ test_normalize_target_concept_id")


def test_normalize_target_special():
    assert normalize_target("manufacturing_overview") == "manufacturing_overview"
    assert normalize_target("MANUFACTURING_OVERVIEW") == "manufacturing_overview"
    print("✓ test_normalize_target_special")


def test_normalize_target_invalid():
    assert normalize_target("") is None
    assert normalize_target("   ") is None
    assert normalize_target("한글만") is None
    print("✓ test_normalize_target_invalid")


def test_normalize_target_list_dedup():
    """중복 제거."""
    normalized, rejected = normalize_target_list(["M1_1", "m1.1", "M1-1"])
    assert normalized == ["M1_1"]
    assert rejected == []
    print("✓ test_normalize_target_list_dedup")


def test_normalize_target_list_mixed():
    """모듈 ID + concept_id 혼재 (옵션 B 핵심)."""
    normalized, rejected = normalize_target_list([
        "M1_1", "M1_4_taylor_principles", "M2_1_2"
    ])
    assert "M1_1" in normalized
    assert "M1_4_taylor_principles" in normalized
    assert "M2_1_2" in normalized
    assert rejected == []
    print("✓ test_normalize_target_list_mixed")


def test_normalize_target_list_rejection():
    """정규화 실패 항목은 rejected에."""
    normalized, rejected = normalize_target_list(["M1_1", "한글", "invalid stuff"])
    assert "M1_1" in normalized
    assert "한글" in rejected
    print("✓ test_normalize_target_list_rejection")


# ────────────────────────────────────────────────────────────
# 2. _input_to_text — 모든 입력 모드
# ────────────────────────────────────────────────────────────
def test_input_to_text_none():
    assert _input_to_text(None) == ""
    print("✓ test_input_to_text_none")


def test_input_to_text_dict():
    text = _input_to_text({"total_points": 100})
    assert "total_points" in text
    assert "100" in text
    print("✓ test_input_to_text_dict")


def test_input_to_text_str():
    assert _input_to_text("Hello professor") == "Hello professor"
    print("✓ test_input_to_text_str")


def test_input_to_text_path_missing():
    with pytest.raises(FileNotFoundError):
        _input_to_text(Path("/nonexistent/file.txt"))
    print("✓ test_input_to_text_path_missing")


def test_input_to_text_path_text(tmp_path):
    p = tmp_path / "req.txt"
    p.write_text("75분 시험", encoding="utf-8")
    assert _input_to_text(p) == "75분 시험"
    print("✓ test_input_to_text_path_text")


def test_input_to_text_path_json(tmp_path):
    p = tmp_path / "req.json"
    p.write_text('{"total_points": 100}', encoding="utf-8")
    result = _input_to_text(p)
    assert "total_points" in result
    print("✓ test_input_to_text_path_json")


def test_input_to_text_unsupported_type():
    with pytest.raises(TypeError):
        _input_to_text(12345)  # int 미지원
    print("✓ test_input_to_text_unsupported_type")


# ────────────────────────────────────────────────────────────
# 3. parse() — None 입력 (Case 1)
# ────────────────────────────────────────────────────────────
def test_parse_none_input_uses_all_defaults():
    """None 입력 → LLM이 all-null 반환 → defaults로 완전한 ReqVector."""
    parser = ReqParser(GeminiClient(mock=True, mock_responder=_all_null_responder()))
    req = parser.parse(None)

    assert isinstance(req, ReqVector)
    assert req.total_points == EXAM_BASICS["total_points"]
    assert req.duration_minutes == EXAM_BASICS["duration_minutes"]
    assert req.target_chapters_or_concepts == list(TARGET_CHAPTERS)
    assert req.evaluation_focus == EvaluationFocus.BALANCED
    # warnings에 defaults 사용 흔적
    assert any("defaults" in w for w in req.warnings)
    print("✓ test_parse_none_input_uses_all_defaults")


# ────────────────────────────────────────────────────────────
# 4. parse() — 완전한 dict (Case 2)
# ────────────────────────────────────────────────────────────
def test_parse_complete_dict_no_defaults_warnings():
    """완전한 dict → LLM이 그대로 반환 → defaults warning 거의 없음."""
    parser = ReqParser(GeminiClient(mock=True, mock_responder=_full_responder()))
    req = parser.parse({"total_points": 100, "duration_minutes": 75})

    assert req.total_points == 100
    assert req.exam_title == "Test Exam"
    assert req.target_chapters_or_concepts == ["M1_1", "M1_2"]
    # exam_title/target 등이 LLM에서 다 채워졌으니 그 필드의 'defaults 사용' warning은 없어야
    title_default_warnings = [w for w in req.warnings if "exam_title: defaults" in w]
    assert title_default_warnings == []
    print("✓ test_parse_complete_dict_no_defaults_warnings")


# ────────────────────────────────────────────────────────────
# 5. parse() — 부분 dict (Case 3)
# ────────────────────────────────────────────────────────────
def test_parse_partial_dict_records_warnings():
    """일부 필드만 → 누락 필드는 defaults + warnings에 어떤 필드 default 됐는지."""
    responder = _make_responder({
        "exam_title": None,
        "exam_type": "midterm",
        "target_chapters_or_concepts": ["M1_1"],
        "total_points": 100,
        "duration_minutes": None,  # 누락
        "evaluation_focus": None,  # 누락
        "difficulty_distribution_group": None,  # 누락
        "question_type_distribution": None,  # 누락
        "allowed_formats": None,
        "special_instructions": None,
        "_warnings": [],
    })
    parser = ReqParser(GeminiClient(mock=True, mock_responder=responder))
    req = parser.parse({"total_points": 100, "target": "M1_1"})

    assert req.total_points == 100
    assert req.duration_minutes == EXAM_BASICS["duration_minutes"]
    # 누락 필드 마다 warning이 있어야
    assert any("duration_minutes: defaults" in w for w in req.warnings)
    assert any("difficulty_distribution_group: defaults" in w for w in req.warnings)
    print("✓ test_parse_partial_dict_records_warnings")


# ────────────────────────────────────────────────────────────
# 6. parse() — 자유 텍스트 입력 (Case 4)
# ────────────────────────────────────────────────────────────
def test_parse_free_text_input(tmp_path):
    """자유 텍스트 (mock LLM) → ReqVector 정상."""
    parser = ReqParser(GeminiClient(mock=True, mock_responder=_full_responder()))
    req = parser.parse("75분 100점 미드텀")

    assert req.total_points == 100
    assert req.exam_type == "midterm"
    print("✓ test_parse_free_text_input")


# ────────────────────────────────────────────────────────────
# 7. parse() — Path 입력 (Case 5)
# ────────────────────────────────────────────────────────────
def test_parse_path_input_text(tmp_path):
    p = tmp_path / "req.txt"
    p.write_text("75분 시험", encoding="utf-8")
    parser = ReqParser(GeminiClient(mock=True, mock_responder=_full_responder()))
    req = parser.parse(p)
    assert isinstance(req, ReqVector)
    print("✓ test_parse_path_input_text")


def test_parse_path_missing_file_falls_back():
    """존재하지 않는 파일 → defaults fallback (raise 안 함)."""
    parser = ReqParser(GeminiClient(mock=True, mock_responder=_full_responder()))
    req = parser.parse(Path("/nonexistent/x.txt"))
    assert isinstance(req, ReqVector)
    assert any("입력 처리 실패" in w for w in req.warnings)
    print("✓ test_parse_path_missing_file_falls_back")


# ────────────────────────────────────────────────────────────
# 8. parse() — LLM 실패 (Case 6)
# ────────────────────────────────────────────────────────────
def test_parse_llm_returns_invalid_json_falls_back():
    """LLM이 깨진 JSON → defaults fallback."""
    def broken_responder(s, u):
        return "this is not json {"
    parser = ReqParser(GeminiClient(mock=True, mock_responder=broken_responder))
    req = parser.parse("anything")
    assert isinstance(req, ReqVector)
    assert any("LLM 파싱 실패" in w for w in req.warnings)
    # defaults가 적용됐는지 확인
    assert req.total_points == EXAM_BASICS["total_points"]
    print("✓ test_parse_llm_returns_invalid_json_falls_back")


def test_parse_llm_returns_wrong_schema_falls_back():
    """LLM이 ParsedRawOutput으로 검증 안 되는 JSON → defaults fallback."""
    def wrong_responder(s, u):
        return json.dumps({"unknown_key": 123, "_warnings": []})
    parser = ReqParser(GeminiClient(mock=True, mock_responder=wrong_responder))
    req = parser.parse("x")
    assert isinstance(req, ReqVector)
    assert any("LLM 파싱 실패" in w for w in req.warnings)
    print("✓ test_parse_llm_returns_wrong_schema_falls_back")


# ────────────────────────────────────────────────────────────
# 9. parse() — 잘못된 enum (Case 7)
# ────────────────────────────────────────────────────────────
def test_parse_invalid_evaluation_focus_falls_back():
    """evaluation_focus가 enum에 없는 값 → defaults("balanced") + warning."""
    responder = _make_responder({
        "exam_title": None, "exam_type": None,
        "target_chapters_or_concepts": None,
        "total_points": None, "duration_minutes": None,
        "evaluation_focus": "garbage_value",
        "difficulty_distribution_group": None,
        "question_type_distribution": None,
        "allowed_formats": None, "special_instructions": None,
        "_warnings": [],
    })
    parser = ReqParser(GeminiClient(mock=True, mock_responder=responder))
    req = parser.parse("x")
    assert req.evaluation_focus == EvaluationFocus.BALANCED
    assert any("evaluation_focus" in w and "garbage_value" in w for w in req.warnings)
    print("✓ test_parse_invalid_evaluation_focus_falls_back")


# ────────────────────────────────────────────────────────────
# 10. parse() — 잘못된 숫자 (Case 8)
# ────────────────────────────────────────────────────────────
def test_parse_negative_total_points_falls_back():
    responder = _make_responder({
        "exam_title": None, "exam_type": None,
        "target_chapters_or_concepts": None,
        "total_points": -50, "duration_minutes": None,
        "evaluation_focus": None,
        "difficulty_distribution_group": None,
        "question_type_distribution": None,
        "allowed_formats": None, "special_instructions": None,
        "_warnings": [],
    })
    parser = ReqParser(GeminiClient(mock=True, mock_responder=responder))
    req = parser.parse("x")
    assert req.total_points == EXAM_BASICS["total_points"]
    assert any("total_points" in w and "-50" in w for w in req.warnings)
    print("✓ test_parse_negative_total_points_falls_back")


# ────────────────────────────────────────────────────────────
# 11. parse() — 합계 불일치 (Case 9)
# ────────────────────────────────────────────────────────────
def test_parse_distribution_sum_mismatch_warns():
    """difficulty 합과 type 합이 다르면 warning."""
    responder = _make_responder({
        "exam_title": None, "exam_type": None,
        "target_chapters_or_concepts": None,
        "total_points": None, "duration_minutes": None,
        "evaluation_focus": None,
        "difficulty_distribution_group": {"easy": 2, "medium": 6, "hard": 2},  # 합 10
        "question_type_distribution": {
            "MCQ_single": 0, "short_answer": 1, "long_answer": 2, "case_analysis": 1,  # 합 4
        },
        "allowed_formats": None, "special_instructions": None,
        "_warnings": [],
    })
    parser = ReqParser(GeminiClient(mock=True, mock_responder=responder))
    req = parser.parse("x")
    assert any("분포 합계 불일치" in w for w in req.warnings)
    print("✓ test_parse_distribution_sum_mismatch_warns")


# ────────────────────────────────────────────────────────────
# 12. parse() — target 정규화 (옵션 B 핵심)
# ────────────────────────────────────────────────────────────
def test_parse_targets_mixed_module_and_concept():
    """모듈 ID와 concept_id가 혼재된 채로 정규화 후 보존."""
    responder = _make_responder({
        "exam_title": None, "exam_type": None,
        "target_chapters_or_concepts": ["m1.1", "M2_1_2_KJ_Method", "M1_2"],
        "total_points": None, "duration_minutes": None,
        "evaluation_focus": None,
        "difficulty_distribution_group": None,
        "question_type_distribution": None,
        "allowed_formats": None, "special_instructions": None,
        "_warnings": [],
    })
    parser = ReqParser(GeminiClient(mock=True, mock_responder=responder))
    req = parser.parse("x")
    # 정규화 결과
    assert "M1_1" in req.target_chapters_or_concepts
    assert "M2_1_2_kj_method" in req.target_chapters_or_concepts
    assert "M1_2" in req.target_chapters_or_concepts
    print("✓ test_parse_targets_mixed_module_and_concept")


def test_parse_targets_empty_uses_defaults():
    responder = _make_responder({
        "exam_title": None, "exam_type": None,
        "target_chapters_or_concepts": [],  # 빈 리스트
        "total_points": None, "duration_minutes": None,
        "evaluation_focus": None,
        "difficulty_distribution_group": None,
        "question_type_distribution": None,
        "allowed_formats": None, "special_instructions": None,
        "_warnings": [],
    })
    parser = ReqParser(GeminiClient(mock=True, mock_responder=responder))
    req = parser.parse("x")
    assert req.target_chapters_or_concepts == list(TARGET_CHAPTERS)
    print("✓ test_parse_targets_empty_uses_defaults")


# ────────────────────────────────────────────────────────────
# 13. LLM 호출 횟수 (Case 16: 모든 입력에 LLM 1회)
# ────────────────────────────────────────────────────────────
def test_parse_always_calls_llm_once():
    """결정 1: 모든 입력에 LLM 1회 호출."""
    reset_usage()
    parser = ReqParser(GeminiClient(mock=True, mock_responder=_full_responder()))

    parser.parse(None)
    parser.parse({"total_points": 100})
    parser.parse("free text")

    records = [r for r in _tracker.records if r.agent_name == "Req_Parser"]
    assert len(records) == 3
    # config/defaults.py에 Req_Parser 명시 없으므로 default light 사용
    from config.defaults import LLM_CONFIG
    light = LLM_CONFIG["models"]["light"]
    for r in records:
        assert light in r.model
    print("✓ test_parse_always_calls_llm_once")


# ────────────────────────────────────────────────────────────
# 14. ReqVector 직렬화 호환성 (Case 17)
# ────────────────────────────────────────────────────────────
def test_parsed_req_vector_serializes():
    parser = ReqParser(GeminiClient(mock=True, mock_responder=_full_responder()))
    req = parser.parse(None)
    serialized = req.model_dump_json()
    rehydrated = ReqVector.model_validate_json(serialized)
    assert rehydrated.exam_title == req.exam_title
    assert rehydrated.target_chapters_or_concepts == req.target_chapters_or_concepts
    print("✓ test_parsed_req_vector_serializes")


# ────────────────────────────────────────────────────────────
# 15. fixture 파일 입력
# ────────────────────────────────────────────────────────────
def test_parse_from_fixture_dict():
    fixture = FIXTURES_DIR / "sample_req_input_dict.json"
    parser = ReqParser(GeminiClient(mock=True, mock_responder=_full_responder()))
    req = parser.parse(fixture)
    assert isinstance(req, ReqVector)
    print("✓ test_parse_from_fixture_dict")


def test_parse_from_fixture_text():
    fixture = FIXTURES_DIR / "sample_req_input_text.txt"
    parser = ReqParser(GeminiClient(mock=True, mock_responder=_full_responder()))
    req = parser.parse(fixture)
    assert isinstance(req, ReqVector)
    print("✓ test_parse_from_fixture_text")


# ────────────────────────────────────────────────────────────
# 16. Smoke: LLM이 _warnings에 모호함 기록 시 전달됨
# ────────────────────────────────────────────────────────────
def test_parse_llm_warnings_propagated():
    responder = _make_responder({
        "exam_title": None, "exam_type": None,
        "target_chapters_or_concepts": None,
        "total_points": None, "duration_minutes": None,
        "evaluation_focus": None,
        "difficulty_distribution_group": None,
        "question_type_distribution": None,
        "allowed_formats": None, "special_instructions": None,
        "_warnings": ["교수가 시험 시간을 75분과 90분으로 두 번 언급함"],
    })
    parser = ReqParser(GeminiClient(mock=True, mock_responder=responder))
    req = parser.parse("75분... 아니 90분")
    assert any("75분과 90분" in w for w in req.warnings)
    print("✓ test_parse_llm_warnings_propagated")


# ────────────────────────────────────────────────────────────
# Runner (pytest 안 쓸 때)
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_normalize_target_module_id()
    test_normalize_target_concept_id()
    test_normalize_target_special()
    test_normalize_target_invalid()
    test_normalize_target_list_dedup()
    test_normalize_target_list_mixed()
    test_normalize_target_list_rejection()
    test_input_to_text_none()
    test_input_to_text_dict()
    test_input_to_text_str()
    print("\n✓ All Req_Parser unit tests passed (non-PDF subset)")
