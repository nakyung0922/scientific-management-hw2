"""
Unit tests for Q&A Generator (mock 모드).
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.qa_generator import (
    AGENT_NAME_HEAVY,
    AGENT_NAME_LIGHT,
    QAGenerator,
    ReworkContext,
    build_user_prompt,
    collect_concepts_for_slot,
    post_process_response,
    select_agent_name,
)
from common.enums import QuestionType, TesterName, Verdict, RoutingStatus
from common.gemini_client import GeminiClient, get_usage_summary, reset_usage
from common.schemas import (
    ConceptKnowledgeStructure,
    ExamBlueprint,
    NextAction,
    Question,
    TesterOutput,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def load_blueprint_and_structure() -> tuple[ExamBlueprint, ConceptKnowledgeStructure]:
    with open(FIXTURES_DIR / "sample_blueprint.json", encoding="utf-8") as f:
        blueprint = ExamBlueprint.model_validate(json.load(f))
    with open(FIXTURES_DIR / "sample_knowledge_structure.json", encoding="utf-8") as f:
        structure = ConceptKnowledgeStructure.model_validate(json.load(f))
    return blueprint, structure


def fake_qa_responder(system_prompt: str, user_prompt: str) -> str:
    """간단한 mock Q&A 응답.

    실제 LLM은 슬롯마다 다른 응답을 주지만, 여기선 *동일한* 구조의 응답을
    반환하면서 user_prompt의 일부를 echo해서 입력이 잘 들어왔는지만 확인.
    """
    # user_prompt에서 slot_id 추출 (간단한 파싱)
    import re
    slot_match = re.search(r'"slot_id":\s*"(Q\d+)"', user_prompt)
    slot_id = slot_match.group(1) if slot_match else "Q??"

    qtype_match = re.search(r'"question_type":\s*"(\w+)"', user_prompt)
    qtype = qtype_match.group(1) if qtype_match else "long_answer"

    # rework 모드인지
    is_rework = '"rework_context":\\s*null' not in user_prompt and "previous_question_id" in user_prompt

    response = {
        "question_id": f"{slot_id}-v1" if not is_rework else f"{slot_id}-v2",
        "prompt": f"[Mock {qtype}] {slot_id} 문항 본문입니다.",
        "choices": None,
        "reference_answer": f"[Mock answer for {slot_id}]",
        "answer_steps": ["단계 1", "단계 2"],
        "source_references": [
            {
                "concept_id": "M1_4_taylor_principles",
                "page_or_slide": "M1.4 p.3",
                "excerpt": "테일러의 과학적 관리 4원칙은 ... (mock 인용)",
            }
        ],
        "warnings": [],
    }
    return json.dumps(response, ensure_ascii=False)


# ────────────────────────────────────────────────────────────
# 결정론적 헬퍼 테스트
# ────────────────────────────────────────────────────────────
def test_select_agent_name():
    blueprint, _ = load_blueprint_and_structure()
    slots_by_id = {s.slot_id: s for s in blueprint.question_slots}

    # case_analysis → heavy
    assert select_agent_name(slots_by_id["Q01"]) == AGENT_NAME_HEAVY
    # long_answer → heavy
    assert select_agent_name(slots_by_id["Q04"]) == AGENT_NAME_HEAVY
    # short_answer → light
    assert select_agent_name(slots_by_id["Q09"]) == AGENT_NAME_LIGHT
    print("✓ select_agent_name")


def test_collect_concepts_for_slot():
    blueprint, structure = load_blueprint_and_structure()
    slot = blueprint.question_slots[0]  # Q01: 3 primary concepts

    concepts = collect_concepts_for_slot(slot, structure)
    assert len(concepts) == 3
    concept_ids = {c.concept_id for c in concepts}
    assert concept_ids == set(slot.primary_concept_ids)
    print("✓ collect_concepts_for_slot")


def test_collect_concepts_missing_id():
    """그래프에 없는 ID는 skip."""
    blueprint, structure = load_blueprint_and_structure()
    slot = blueprint.question_slots[0].model_copy()
    slot.primary_concept_ids = list(slot.primary_concept_ids) + ["NONEXISTENT_ID"]

    concepts = collect_concepts_for_slot(slot, structure)
    # NONEXISTENT_ID는 skip되어야 함
    assert all(c.concept_id != "NONEXISTENT_ID" for c in concepts)
    print("✓ collect_concepts_missing_id (skip 처리)")


def test_build_user_prompt_structure():
    blueprint, structure = load_blueprint_and_structure()
    slot = blueprint.question_slots[0]
    concepts = collect_concepts_for_slot(slot, structure)

    prompt = build_user_prompt(
        slot, concepts, session_id="test-session"
    )

    assert "test-session" in prompt
    assert slot.slot_id in prompt
    assert '"rework_context": null' in prompt
    # concepts 정보가 들어갔는지
    for c in concepts:
        assert c.concept_id in prompt
    print("✓ build_user_prompt_structure")


def test_build_user_prompt_rework():
    blueprint, structure = load_blueprint_and_structure()
    slot = blueprint.question_slots[0]
    concepts = collect_concepts_for_slot(slot, structure)

    fake_prev = Question(
        question_id="Q01-v1",
        slot_id="Q01",
        session_id="test-session",
        question_type=QuestionType.CASE_ANALYSIS,
        target_difficulty_level=5,
        points=7,
        prompt="이전 문항",
        reference_answer="이전 답안",
        answer_steps=[],
        source_references=[],
        generation_metadata={
            "generator_version": "qa_gen_v0.1",
            "attempt_number": 1,
            "previous_question_id": None,
        },
    )

    rework = ReworkContext(
        previous_question=fake_prev,
        failure_reason="강의자료를 벗어난 사례 사용",
    )
    prompt = build_user_prompt(
        slot, concepts, session_id="test-session", rework_context=rework
    )

    assert "강의자료를 벗어난" in prompt
    assert "Q01-v1" in prompt
    print("✓ build_user_prompt_rework")


# ────────────────────────────────────────────────────────────
# 통합 테스트 (mock LLM)
# ────────────────────────────────────────────────────────────
def test_generate_one():
    """단일 슬롯에 대해 Question 생성."""
    blueprint, structure = load_blueprint_and_structure()
    slot = blueprint.question_slots[3]  # Q04: long_answer L3

    client = GeminiClient(mock=True, mock_responder=fake_qa_responder)
    gen = QAGenerator(gemini_client=client)

    question = gen.generate_one(slot, structure, blueprint.session_id)

    # 스키마 검증은 Pydantic이 함. 강제 주입된 메타는 그대로인지.
    assert question.slot_id == slot.slot_id
    assert question.session_id == blueprint.session_id
    assert int(question.target_difficulty_level) == int(slot.target_difficulty_level)
    assert question.points == slot.points
    assert question.generation_metadata.attempt_number == 1
    assert question.generation_metadata.previous_question_id is None
    assert question.generation_metadata.generator_version == "qa_gen_v0.1"
    print(f"✓ generate_one ({slot.slot_id}) → {question.question_id}")


def test_generate_all():
    """Blueprint의 모든 슬롯에 대해 Question 생성."""
    blueprint, structure = load_blueprint_and_structure()
    reset_usage()

    client = GeminiClient(mock=True, mock_responder=fake_qa_responder)
    gen = QAGenerator(gemini_client=client)

    questions = gen.generate_all(blueprint, structure)

    assert len(questions) == 10
    # 모두 다른 slot_id
    slot_ids = {q.slot_id for q in questions}
    assert len(slot_ids) == 10

    # 모두 mock_responder의 standard 응답 패턴 따름
    for q in questions:
        assert q.session_id == blueprint.session_id
        assert q.reference_answer.startswith("[Mock answer")

    # 사용량 추적 — 10번 호출
    summary = get_usage_summary()
    assert summary["total_calls"] == 10
    # heavy/light 비율 확인
    heavy_calls = summary["per_agent"].get(AGENT_NAME_HEAVY, {}).get("calls", 0)
    light_calls = summary["per_agent"].get(AGENT_NAME_LIGHT, {}).get("calls", 0)
    assert heavy_calls + light_calls == 10
    # case_analysis 3 + long_answer 5 = 8 heavy, short_answer 2 = 2 light
    assert heavy_calls == 8
    assert light_calls == 2
    print(f"✓ generate_all → {len(questions)} questions "
          f"(heavy={heavy_calls}, light={light_calls})")


def test_regenerate_from_feedback():
    """Question Rework 루프."""
    blueprint, structure = load_blueprint_and_structure()
    slot = blueprint.question_slots[3]

    client = GeminiClient(mock=True, mock_responder=fake_qa_responder)
    gen = QAGenerator(gemini_client=client)

    # 1차 생성
    q1 = gen.generate_one(slot, structure, blueprint.session_id)

    # Fail 시뮬레이션
    tester_output = TesterOutput(
        tester_name=TesterName.FACTFULNESS,
        question_id=q1.question_id,
        session_id=blueprint.session_id,
        score=0.5,
        verdict=Verdict.FAIL,
        threshold_used=0.8,
        reasons=["강의자료에 없는 인물을 사실인 양 인용함"],
        next_action=NextAction(
            routing_status=RoutingStatus.QUESTION_REWORK,
            target_agent="QA_Generator",
        ),
    )

    # 재생성
    q2 = gen.regenerate_from_feedback(q1, slot, structure, tester_output)

    # attempt_number 증가, previous_question_id 기록
    assert q2.generation_metadata.attempt_number == 2
    assert q2.generation_metadata.previous_question_id == q1.question_id
    assert q2.question_id != q1.question_id
    print(f"✓ regenerate_from_feedback: {q1.question_id} → {q2.question_id}")


def test_question_serialization():
    """Question round-trip."""
    blueprint, structure = load_blueprint_and_structure()
    slot = blueprint.question_slots[0]

    client = GeminiClient(mock=True, mock_responder=fake_qa_responder)
    gen = QAGenerator(gemini_client=client)
    q = gen.generate_one(slot, structure, blueprint.session_id)

    serialized = q.model_dump_json()
    rehydrated = Question.model_validate_json(serialized)
    assert rehydrated.question_id == q.question_id
    assert rehydrated.slot_id == q.slot_id
    print("✓ question_serialization")


# ────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_select_agent_name()
    test_collect_concepts_for_slot()
    test_collect_concepts_missing_id()
    test_build_user_prompt_structure()
    test_build_user_prompt_rework()
    test_generate_one()
    test_generate_all()
    test_regenerate_from_feedback()
    test_question_serialization()
    print("\n✓ All Q&A Generator tests passed")
