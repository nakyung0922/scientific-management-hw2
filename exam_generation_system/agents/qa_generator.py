"""
Q&A Generator Agent — Agent 3-2 (PG2, R4).

입력:
- ExamBlueprint (Planner 출력)
- ConceptKnowledgeStructure (concept 상세 정보)
- (선택) rework_context: 재생성 모드

출력:
- 슬롯마다 Question 객체 1개씩

설계:
- 슬롯별 독립 호출 (병렬화 가능)
- heavy/light 모델 자동 선택
  - case_analysis, long_answer → heavy
  - short_answer, MCQ_single → light
- 결정론: 메타데이터 채우기, ID 부여, source_references 후처리
- LLM: 문항 텍스트, 모범답안, answer_steps 생성
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from common.concept_utils import node_to_llm_context
from common.enums import QuestionType
from common.gemini_client import GeminiClient
from common.schemas import (
    ConceptKnowledgeStructure,
    ConceptNode,
    ExamBlueprint,
    GenerationMetadata,
    Question,
    QuestionSlot,
    SourceReference,
    TesterOutput,
)
from config.defaults import PROMPTS_DIR

logger = logging.getLogger(__name__)

AGENT_NAME_HEAVY = "QA_Generator_heavy"
AGENT_NAME_LIGHT = "QA_Generator_light"
GENERATOR_VERSION = "qa_gen_v0.1"
PROMPT_PATH = PROMPTS_DIR / "qa_generator.md"


# 모델 자동 선택용 매핑
HEAVY_TYPES = {QuestionType.LONG_ANSWER, QuestionType.CASE_ANALYSIS}


# ────────────────────────────────────────────────────────────
# Rework context (Question Rework 모드용)
# ────────────────────────────────────────────────────────────
class ReworkContext:
    """이전 시도 정보 + 실패 사유."""

    def __init__(
        self,
        previous_question: Question,
        failure_reason: str,
    ):
        self.previous_question = previous_question
        self.failure_reason = failure_reason

    def to_dict(self) -> dict:
        return {
            "previous_question_id": self.previous_question.question_id,
            "previous_prompt": self.previous_question.prompt,
            "previous_reference_answer": self.previous_question.reference_answer,
            "failure_reason": self.failure_reason,
        }


# ────────────────────────────────────────────────────────────
# Prompt 빌더
# ────────────────────────────────────────────────────────────
def build_user_prompt(
    slot: QuestionSlot,
    concepts: list[ConceptNode],
    session_id: str,
    rework_context: Optional[ReworkContext] = None,
) -> str:
    """슬롯 + concept 정보 → user prompt."""
    slot_dict = slot.model_dump()

    concepts_dicts = [
        node_to_llm_context(c, include_embedding=False)
        for c in concepts
    ]

    payload = {
        "session_id": session_id,
        "slot": slot_dict,
        "concepts": concepts_dicts,
        "rework_context": rework_context.to_dict() if rework_context else None,
    }

    instruction = (
        "다음 슬롯에 대해 시험 문항과 모범답안을 생성하세요. "
        "system_prompt의 모든 규칙을 준수하고, 명시된 JSON 스키마로만 출력하세요."
    )

    return f"{instruction}\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


# ────────────────────────────────────────────────────────────
# concept 수집 헬퍼
# ────────────────────────────────────────────────────────────
def collect_concepts_for_slot(
    slot: QuestionSlot,
    structure: ConceptKnowledgeStructure,
) -> list[ConceptNode]:
    """슬롯의 primary + secondary concept_id에 해당하는 노드들 수집.

    구조에 없는 ID는 warning 로깅 후 skip.
    """
    all_ids = list(slot.primary_concept_ids) + list(slot.secondary_concept_ids)
    concepts: list[ConceptNode] = []
    for cid in all_ids:
        node = structure.get_node(cid)
        if node is None:
            logger.warning(
                f"[{slot.slot_id}] concept_id '{cid}'를 graph에서 찾을 수 없음"
            )
            continue
        concepts.append(node)
    return concepts


# ────────────────────────────────────────────────────────────
# 모델 선택
# ────────────────────────────────────────────────────────────
def select_agent_name(slot: QuestionSlot) -> str:
    """슬롯의 유형에 따라 heavy/light agent 이름 결정.

    heavy 모델: long_answer, case_analysis
    light 모델: short_answer, MCQ_single
    """
    qtype = (
        QuestionType(slot.question_type)
        if isinstance(slot.question_type, str)
        else slot.question_type
    )
    return AGENT_NAME_HEAVY if qtype in HEAVY_TYPES else AGENT_NAME_LIGHT


# ────────────────────────────────────────────────────────────
# 결과 후처리
# ────────────────────────────────────────────────────────────
def post_process_response(
    llm_response: dict,
    slot: QuestionSlot,
    session_id: str,
    rework_context: Optional[ReworkContext],
) -> Question:
    """LLM 응답 dict → Question Pydantic 객체로 변환.

    - 메타데이터 강제 주입 (session_id, generator_version 등)
    - 누락 필드 기본값 보강
    - Pydantic 검증으로 스키마 위반 즉시 발견
    """
    # 메타데이터
    attempt = 1
    prev_id = None
    if rework_context:
        attempt = (
            rework_context.previous_question.generation_metadata.attempt_number + 1
        )
        prev_id = rework_context.previous_question.question_id

    # LLM 응답을 신뢰하되, 핵심 필드는 슬롯에서 강제 덮어쓰기
    enforced = {
        "slot_id": slot.slot_id,
        "session_id": session_id,
        "question_type": (
            slot.question_type.value
            if hasattr(slot.question_type, "value")
            else slot.question_type
        ),
        "target_difficulty_level": int(slot.target_difficulty_level),
        "points": slot.points,
        "generation_metadata": GenerationMetadata(
            generator_version=GENERATOR_VERSION,
            attempt_number=attempt,
            previous_question_id=prev_id,
        ).model_dump(),
    }

    merged = {**llm_response, **enforced}

    # question_id: LLM이 빠뜨리면 slot_id + 버전으로 채움
    if "question_id" not in merged or not merged["question_id"]:
        merged["question_id"] = f"{slot.slot_id}-v{attempt}"

    # source_references 누락 시 빈 배열 (검증으로 잡음)
    merged.setdefault("source_references", [])
    merged.setdefault("answer_steps", [])
    merged.setdefault("warnings", [])

    # choices가 MCQ가 아닌데 들어있으면 정리
    qtype_str = merged["question_type"]
    if qtype_str != "MCQ_single":
        merged["choices"] = None

    return Question.model_validate(merged)


# ────────────────────────────────────────────────────────────
# 메인 Generator 클래스
# ────────────────────────────────────────────────────────────
class QAGenerator:
    """Q&A Generator Agent.

    Usage:
        gen = QAGenerator(gemini_client=GeminiClient(mock=True))
        questions = gen.generate_all(blueprint, knowledge_structure)
    """

    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def generate_one(
        self,
        slot: QuestionSlot,
        structure: ConceptKnowledgeStructure,
        session_id: str,
        *,
        rework_context: Optional[ReworkContext] = None,
    ) -> Question:
        """단일 슬롯에 대해 Question 생성. (rework 모드 지원)"""
        concepts = collect_concepts_for_slot(slot, structure)
        if not concepts:
            raise ValueError(
                f"[{slot.slot_id}] 슬롯에 유효한 concept이 하나도 없습니다. "
                f"primary_concept_ids={slot.primary_concept_ids}"
            )

        user_prompt = build_user_prompt(
            slot, concepts, session_id, rework_context
        )
        agent_name = select_agent_name(slot)

        llm_response = self.client.generate_json(
            agent_name=agent_name,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,  # 약간의 다양성 허용 (재생성 시 다른 답 가능)
        )

        question = post_process_response(
            llm_response, slot, session_id, rework_context
        )
        return question

    def generate_all(
        self,
        blueprint: ExamBlueprint,
        structure: ConceptKnowledgeStructure,
    ) -> list[Question]:
        """Blueprint의 모든 슬롯에 대해 순차적으로 Question 생성.

        병렬 처리는 향후 도입 (현재는 단순 순회 — 디버깅 쉬움).
        """
        questions: list[Question] = []
        for slot in blueprint.question_slots:
            try:
                q = self.generate_one(
                    slot, structure, blueprint.session_id
                )
                questions.append(q)
            except Exception as e:
                logger.error(f"[{slot.slot_id}] 생성 실패: {e}")
                raise
        return questions

    def regenerate_from_feedback(
        self,
        previous: Question,
        slot: QuestionSlot,
        structure: ConceptKnowledgeStructure,
        tester_output: TesterOutput,
    ) -> Question:
        """Tester 피드백을 받아 재생성 (Question Rework 루프).

        Supervisor가 호출하는 엔트리포인트.
        """
        failure_reason = "; ".join(tester_output.reasons) if tester_output.reasons else \
            f"Tester {tester_output.tester_name} verdict=fail"
        rework_context = ReworkContext(
            previous_question=previous,
            failure_reason=failure_reason,
        )
        return self.generate_one(
            slot, structure, previous.session_id,
            rework_context=rework_context,
        )
