"""
Supervisor end-to-end 통합 테스트.

GCP 없이 mock 모드로 전체 파이프라인 흐름 검증.
실행: python -m pytest tests/test_supervisor.py -v
또는: python tests/test_supervisor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.enums import DifficultyLevel, Importance, QuestionType, RoutingStatus
from common.gemini_client import GeminiClient, reset_usage
from common.schemas import (
    AnswerRubric,
    ConceptEdge,
    ConceptKnowledgeStructure,
    ConceptNode,
    DifficultyDistributionGroup,
    ExamBlueprint,
    ExamMeta,
    DifficultyPolicy,
    GenerationMetadata,
    NextAction,
    QAPaperOutput,
    Question,
    QuestionSlot,
    QuestionTypeDistribution,
    ReqVector,
    RubricCriterion,
    SectionStructure,
    TesterOutput,
)
from common.enums import TesterName, Verdict
from agents.supervisor import Supervisor, measure_automation_rate


# ────────────────────────────────────────────────────────────
# 공통 픽스처
# ────────────────────────────────────────────────────────────
SESSION_ID = "test-session-001"

def make_knowledge_structure() -> ConceptKnowledgeStructure:
    return ConceptKnowledgeStructure(
        nodes=[
            ConceptNode(
                concept_id="M1_1_scientific_management",
                concept_name="Scientific Management",
                depth_in_tree=0,
                importance=Importance.HIGH,
                intrinsic_difficulty_level=DifficultyLevel.L3,
                suitable_question_types=[QuestionType.LONG_ANSWER, QuestionType.CASE_ANALYSIS],
                source_pages=["M1.1 p.1"],
            ),
            ConceptNode(
                concept_id="M1_3_therbligs",
                concept_name="Therbligs",
                depth_in_tree=1,
                parent_concept_id="M1_1_scientific_management",
                importance=Importance.MEDIUM,
                intrinsic_difficulty_level=DifficultyLevel.L1,
                suitable_question_types=[QuestionType.SHORT_ANSWER],
                source_pages=["M1.3 p.5"],
            ),
        ],
        edges=[
            ConceptEdge(
                from_concept_id="M1_1_scientific_management",
                to_concept_id="M1_3_therbligs",
                relation="parent_of",
                weight=1.0,
            ),
        ],
    )


def make_req_vector() -> ReqVector:
    return ReqVector(
        exam_title="Test Midterm",
        exam_type="midterm",
        target_chapters_or_concepts=["M1_1", "M1_3"],
        total_points=100,
        duration_minutes=75,
        difficulty_distribution_group=DifficultyDistributionGroup(
            easy=1, medium=1, hard=0
        ),
        question_type_distribution=QuestionTypeDistribution(
            short_answer=1, long_answer=1
        ),
    )


def make_stub_blueprint(session_id: str) -> ExamBlueprint:
    """최소 2슬롯 blueprint."""
    from common.schemas import RevisionEntry
    return ExamBlueprint(
        session_id=session_id,
        exam_meta=ExamMeta(
            exam_title="Test Midterm",
            exam_type="midterm",
            total_points=100,
            duration_minutes=75,
            target_chapters_or_concepts=["M1_1", "M1_3"],
            evaluation_focus="balanced",
            weights_applied={"alpha": 0.5, "beta": 0.5},
        ),
        difficulty_policy=DifficultyPolicy(
            target_distribution_group=DifficultyDistributionGroup(
                easy=1, medium=1, hard=0
            ),
            target_distribution_level={"L1": 1, "L3": 1},
        ),
        section_structure=[
            SectionStructure(
                section_id="S1",
                section_title="단답형",
                question_type=QuestionType.SHORT_ANSWER,
                num_questions=1,
                points_per_question=10,
            ),
            SectionStructure(
                section_id="S2",
                section_title="서술형",
                question_type=QuestionType.LONG_ANSWER,
                num_questions=1,
                points_per_question=12,
            ),
        ],
        question_slots=[
            QuestionSlot(
                slot_id="S01",
                section_id="S1",
                question_type=QuestionType.SHORT_ANSWER,
                target_difficulty_level=DifficultyLevel.L1,
                expected_X1=0.0,
                expected_X2=0.4,
                expected_D=0.2,
                points=10,
                primary_concept_ids=["M1_3_therbligs"],
            ),
            QuestionSlot(
                slot_id="S02",
                section_id="S2",
                question_type=QuestionType.LONG_ANSWER,
                target_difficulty_level=DifficultyLevel.L3,
                expected_X1=0.5,
                expected_X2=0.7,
                expected_D=0.6,
                points=12,
                primary_concept_ids=["M1_1_scientific_management"],
            ),
        ],
    )


def make_stub_question(slot_id: str, session_id: str, attempt: int = 1) -> Question:
    from common.schemas import SourceReference
    return Question(
        question_id=f"Q_{slot_id}_{attempt}",
        slot_id=slot_id,
        session_id=session_id,
        question_type=QuestionType.SHORT_ANSWER,
        target_difficulty_level=DifficultyLevel.L1,
        points=10,
        prompt=f"[stub] {slot_id} 문항입니다.",
        reference_answer="[stub] 모범답안",
        source_references=[
            SourceReference(
                concept_id="M1_3_therbligs",
                page_or_slide="M1.3 p.5",
                excerpt="Therbligs는 ...",
            )
        ],
        generation_metadata=GenerationMetadata(
            generator_version="stub",
            attempt_number=attempt,
        ),
    )


def make_stub_tester_output(
    question: Question,
    tester_name: TesterName,
    score: float,
    threshold: float,
    routing: RoutingStatus = RoutingStatus.FLOW,
    target: str = "Supervisor",
) -> TesterOutput:
    verdict = Verdict.PASS if score >= threshold else Verdict.FAIL
    next_action = None if verdict == Verdict.PASS else NextAction(
        routing_status=routing, target_agent=target
    )
    return TesterOutput(
        tester_name=tester_name,
        question_id=question.question_id,
        session_id=question.session_id,
        score=score,
        verdict=verdict,
        threshold_used=threshold,
        next_action=next_action,
    )


def make_stub_rubric(question: Question) -> AnswerRubric:
    return AnswerRubric(
        question_id=question.question_id,
        session_id=question.session_id,
        total_points=question.points,
        criteria=[
            RubricCriterion(
                criterion_id="C1",
                description="핵심 키워드 포함",
                points=question.points,
                key_points=["키워드1"],
            )
        ],
    )


def make_stub_paper(session_id: str) -> QAPaperOutput:
    return QAPaperOutput(
        session_id=session_id,
        exam_paper_file="output/exam_paper.docx",
        answer_key_file="output/answer_key.docx",
    )


# ────────────────────────────────────────────────────────────
# 테스트 1: 정상 흐름 (모든 슬롯 1회 통과)
# ────────────────────────────────────────────────────────────
def test_happy_path():
    """모든 슬롯이 첫 번째 시도에 통과하는 정상 케이스."""
    print("\n" + "=" * 50)
    print("TEST 1: Happy Path (모든 슬롯 1회 통과)")
    print("=" * 50)

    reset_usage()
    client = GeminiClient(mock=True)
    supervisor = Supervisor(client)

    knowledge_structure = make_knowledge_structure()
    blueprint = make_stub_blueprint(SESSION_ID)

    # 각 에이전트 mock
    stub_q_s01 = make_stub_question("S01", SESSION_ID, 1)
    stub_q_s02 = make_stub_question("S02", SESSION_ID, 1)

    call_count = {"s01": 0, "s02": 0}

    def mock_generate_one(slot, knowledge_structure, session_id, attempt_number, previous_question_id=None):
        if slot.slot_id == "S01":
            call_count["s01"] += 1
            return stub_q_s01
        else:
            call_count["s02"] += 1
            return stub_q_s02

    def mock_factful_test(question, slot, knowledge_structure, session_id):
        return make_stub_tester_output(
            question, TesterName.FACTFULNESS,
            score=0.92, threshold=0.80,
        )

    def mock_diff_test(question, slot, knowledge_structure, session_id):
        return make_stub_tester_output(
            question, TesterName.DIFFICULTY,
            score=0.75, threshold=0.60,
        )

    def mock_rubric_generate(question, slot, knowledge_structure, session_id, evaluation_focus=None):
        return make_stub_rubric(question)

    def mock_format(blueprint, questions, rubrics, session_id):
        return make_stub_paper(session_id)

    supervisor.qa_generator.generate_one    = mock_generate_one
    supervisor.factful_tester.test          = mock_factful_test
    supervisor.diff_tester.test             = mock_diff_test
    supervisor.rubric_machine.generate      = mock_rubric_generate
    supervisor.paper_formattor.format       = mock_format

    # ExamPlanner mock
    supervisor.exam_planner.plan = lambda req, structure, session_id, previous_blueprint=None, correction_reason=None: blueprint
    # adjust_remaining_slots 없음 — plan()의 previous_blueprint 파라미터로 처리

    # _run_step1_parallel stub override
    def mock_step1(pdf_paths, requirements, session_id):
        return knowledge_structure, make_req_vector()

    supervisor._run_step1_parallel = mock_step1

    result = supervisor.run(
        pdf_paths=["M1_1.pdf"],
        requirements="Test requirements",
        hitl=False,
    )

    # 검증
    assert result.status == "COMPLETED", f"Expected COMPLETED, got {result.status}"
    assert len(result.slot_results) == 2
    assert result.slot_results[0].attempts == 1
    assert result.slot_results[1].attempts == 1
    assert not any(sr.exceeded_max_retry for sr in result.slot_results)
    assert result.paper_output is not None
    assert call_count["s01"] == 1
    assert call_count["s02"] == 1

    rate = measure_automation_rate(result)
    assert rate["meets_80_percent"], f"자동화율 80% 미달: {rate}"

    print(f"  ✅ PASSED | 소요: {result.elapsed_sec}초")
    print(f"  📊 자동화율: {rate['step_auto_rate']}% (80% 기준 {'✅' if rate['meets_80_percent'] else '❌'})")
    return True


# ────────────────────────────────────────────────────────────
# 테스트 2: Question Rework (Factfulness 실패 → 재시도)
# ────────────────────────────────────────────────────────────
def test_question_rework():
    """Factfulness 실패 2회 → 3번째 시도에 통과."""
    print("\n" + "=" * 50)
    print("TEST 2: Question Rework (Factfulness 2회 실패 → 3번째 통과)")
    print("=" * 50)

    reset_usage()
    client = GeminiClient(mock=True)
    supervisor = Supervisor(client)

    knowledge_structure = make_knowledge_structure()
    blueprint = make_stub_blueprint(SESSION_ID)

    attempt_tracker = {"count": 0}

    def mock_generate_one(slot, knowledge_structure, session_id, attempt_number, previous_question_id=None):
        attempt_tracker["count"] += 1
        return make_stub_question(slot.slot_id, session_id, attempt_number)

    def mock_factful_test(question, slot, knowledge_structure, session_id):
        # 처음 2번 실패, 3번째부터 통과
        attempt = int(question.question_id.split("_")[-1])
        score   = 0.5 if attempt < 3 else 0.92
        return make_stub_tester_output(
            question, TesterName.FACTFULNESS,
            score=score, threshold=0.80,
            routing=RoutingStatus.QUESTION_REWORK,
            target="QA_Generator",
        )

    def mock_diff_test(question, slot, knowledge_structure, session_id):
        return make_stub_tester_output(
            question, TesterName.DIFFICULTY,
            score=0.75, threshold=0.60,
        )

    supervisor.qa_generator.generate_one = mock_generate_one
    supervisor.factful_tester.test        = mock_factful_test
    supervisor.diff_tester.test           = mock_diff_test
    supervisor.rubric_machine.generate    = lambda question, slot, knowledge_structure, session_id, evaluation_focus=None: make_stub_rubric(question)
    supervisor.paper_formattor.format     = lambda blueprint, questions, rubrics, session_id: make_stub_paper(session_id)
    supervisor.exam_planner.plan          = lambda req, structure, session_id, previous_blueprint=None, correction_reason=None: blueprint
    # adjust_remaining_slots 없음 — plan()의 previous_blueprint 파라미터로 처리
    supervisor._run_step1_parallel = lambda pdf, req, sid: (knowledge_structure, make_req_vector())

    result = supervisor.run(
        pdf_paths=["M1_1.pdf"],
        requirements="Test",
        hitl=False,
    )

    s01_result = result.slot_results[0]
    assert result.status == "COMPLETED"
    assert s01_result.attempts == 3, f"Expected 3 attempts, got {s01_result.attempts}"
    assert not s01_result.exceeded_max_retry

    print(f"  ✅ PASSED | S01 시도 횟수: {s01_result.attempts}회")
    return True


# ────────────────────────────────────────────────────────────
# 테스트 3: MAX_RETRY 초과 → warnings 기록
# ────────────────────────────────────────────────────────────
def test_max_retry_exceeded():
    """3회 모두 Factfulness 실패 → warnings 기록 후 통과."""
    print("\n" + "=" * 50)
    print("TEST 3: MAX_RETRY 초과 → warnings 기록")
    print("=" * 50)

    reset_usage()
    client = GeminiClient(mock=True)
    supervisor = Supervisor(client)

    knowledge_structure = make_knowledge_structure()
    blueprint = make_stub_blueprint(SESSION_ID)

    def mock_generate_one(slot, knowledge_structure, session_id, attempt_number, previous_question_id=None):
        return make_stub_question(slot.slot_id, session_id, attempt_number)

    def mock_factful_test_always_fail(question, slot, knowledge_structure, session_id):
        return make_stub_tester_output(
            question, TesterName.FACTFULNESS,
            score=0.3, threshold=0.80,
            routing=RoutingStatus.QUESTION_REWORK,
            target="QA_Generator",
        )

    supervisor.qa_generator.generate_one = mock_generate_one
    supervisor.factful_tester.test        = mock_factful_test_always_fail
    supervisor.diff_tester.test           = lambda **kwargs: None  # 도달 안 함
    supervisor.rubric_machine.generate    = lambda question, slot, knowledge_structure, session_id, evaluation_focus=None: make_stub_rubric(question)
    supervisor.paper_formattor.format     = lambda blueprint, questions, rubrics, session_id: make_stub_paper(session_id)
    supervisor.exam_planner.plan          = lambda req, structure, session_id, previous_blueprint=None, correction_reason=None: blueprint
    # adjust_remaining_slots 없음 — plan()의 previous_blueprint 파라미터로 처리
    supervisor._run_step1_parallel = lambda pdf, req, sid: (knowledge_structure, make_req_vector())

    result = supervisor.run(
        pdf_paths=["M1_1.pdf"],
        requirements="Test",
        hitl=False,
    )

    s01_result = result.slot_results[0]
    assert result.status == "COMPLETED"
    assert s01_result.exceeded_max_retry, "MAX_RETRY 초과 표시 없음"
    assert s01_result.attempts == 3
    assert len(result.warnings) > 0, "warnings가 기록되지 않음"

    print(f"  ✅ PASSED | warnings: {result.warnings}")
    return True


# ────────────────────────────────────────────────────────────
# 테스트 4: Difficulty Correction 라우팅
# ────────────────────────────────────────────────────────────
def test_difficulty_correction_routing():
    """Difficulty 실패 → difficulty_correction → Planner 후속 슬롯 조정."""
    print("\n" + "=" * 50)
    print("TEST 4: Difficulty Correction 라우팅")
    print("=" * 50)

    reset_usage()
    client = GeminiClient(mock=True)
    supervisor = Supervisor(client)

    knowledge_structure = make_knowledge_structure()
    blueprint = make_stub_blueprint(SESSION_ID)

    adjust_called = {"count": 0}

    def mock_plan(req, structure, session_id, previous_blueprint=None, correction_reason=None):
        if previous_blueprint is not None:
            adjust_called["count"] += 1
        return blueprint

    def mock_factful_pass(question, slot, knowledge_structure, session_id):
        return make_stub_tester_output(
            question, TesterName.FACTFULNESS, score=0.92, threshold=0.80
        )

    slot_attempt = {"S01": 0}

    def mock_generate_one(slot, knowledge_structure, session_id, attempt_number, previous_question_id=None):
        return make_stub_question(slot.slot_id, session_id, attempt_number)

    def mock_diff_test(question, slot, knowledge_structure, session_id):
        # S01 슬롯: Difficulty 실패 → correction
        if slot.slot_id == "S01":
            return make_stub_tester_output(
                question, TesterName.DIFFICULTY,
                score=0.3, threshold=0.60,
                routing=RoutingStatus.DIFFICULTY_CORRECTION,
                target="Exam_Planner",
            )
        return make_stub_tester_output(
            question, TesterName.DIFFICULTY, score=0.75, threshold=0.60
        )

    supervisor.qa_generator.generate_one = mock_generate_one
    supervisor.factful_tester.test        = mock_factful_pass
    supervisor.diff_tester.test           = mock_diff_test
    supervisor.rubric_machine.generate    = lambda question, slot, knowledge_structure, session_id, evaluation_focus=None: make_stub_rubric(question)
    supervisor.paper_formattor.format     = lambda blueprint, questions, rubrics, session_id: make_stub_paper(session_id)
    supervisor.exam_planner.plan          = mock_plan
    supervisor._run_step1_parallel = lambda pdf, req, sid: (knowledge_structure, make_req_vector())

    result = supervisor.run(
        pdf_paths=["M1_1.pdf"],
        requirements="Test",
        hitl=False,
    )

    assert result.status == "COMPLETED"
    assert adjust_called["count"] >= 1, "plan(previous_blueprint=...)가 호출되지 않음"
    s01 = result.slot_results[0]
    assert any("difficulty_correction" in w for w in s01.warnings), \
        "difficulty_correction warning이 없음"

    print(f"  ✅ PASSED | Planner 재조정 호출: {adjust_called['count']}회")
    return True


# ────────────────────────────────────────────────────────────
# 테스트 5: 자동화율 측정
# ────────────────────────────────────────────────────────────
def test_automation_rate():
    """자동화율이 80% 이상인지 확인."""
    print("\n" + "=" * 50)
    print("TEST 5: 자동화율 80% 이상 확인")
    print("=" * 50)

    from agents.supervisor import PipelineResult, SlotResult

    result = PipelineResult(
        session_id="test",
        status="COMPLETED",
        slot_results=[
            SlotResult(slot_id="S01", attempts=1, exceeded_max_retry=False),
            SlotResult(slot_id="S02", attempts=1, exceeded_max_retry=False),
            SlotResult(slot_id="S03", attempts=2, exceeded_max_retry=False),
            SlotResult(slot_id="S04", attempts=3, exceeded_max_retry=True),  # 1개 초과
        ],
        hitl_approved=True,
    )

    rate = measure_automation_rate(result)
    print(f"  slot_auto_rate : {rate['slot_auto_rate']}%")
    print(f"  step_auto_rate : {rate['step_auto_rate']}%")
    print(f"  meets_80%      : {rate['meets_80_percent']}")

    assert rate["meets_80_percent"], f"자동화율 80% 미달: {rate['step_auto_rate']}%"
    assert rate["slot_auto_rate"] == 75.0  # 3/4 슬롯 자동
    assert rate["step_auto_rate"] == round(6/7*100, 1)  # 6/7 STEP 자동

    print(f"  ✅ PASSED")
    return True


# ────────────────────────────────────────────────────────────
# 실행
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("Happy Path",               test_happy_path),
        ("Question Rework",          test_question_rework),
        ("MAX_RETRY 초과",           test_max_retry_exceeded),
        ("Difficulty Correction",    test_difficulty_correction_routing),
        ("자동화율 측정",             test_automation_rate),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            ok = test_fn()
            if ok:
                passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"테스트 결과: {passed}/{len(tests)} 통과")
    if errors:
        print("실패 목록:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
