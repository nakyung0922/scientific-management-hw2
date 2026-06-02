"""
Exam Planner Agent — Agent 3-1 (PG2, R4).

입력: ReqVector + ConceptKnowledgeStructure
출력: ExamBlueprint

설계 원칙:
- 결정론적 로직(분포 세분화, X₁/X₂/D 계산, 검증)은 Python으로 직접 처리.
- LLM은 "어떤 concept이 슬롯에 적합한가" 같은 의미적 판단에만 사용.
- 이로써 정량 규칙 위반 가능성 ↓ + 토큰 비용 ↓.
"""
from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from common.concept_utils import (
    ROOT_CONCEPT_ID,
    get_assignable_nodes,
    get_max_distance_in_subset,
    is_root_node,
    node_to_llm_context,
)
from common.difficulty import (
    compute_d,
    compute_x1,
    compute_x2,
    d_to_level,
    get_weights,
    is_valid_type_level_combo,
)
from common.enums import (
    DifficultyGroup,
    DifficultyLevel,
    EvaluationFocus,
    QuestionType,
    RoutingStatus,
)
from common.gemini_client import GeminiClient
from common.schemas import (
    ConceptKnowledgeStructure,
    ConceptNode,
    DifficultyDistributionGroup,
    DifficultyPolicy,
    ExamBlueprint,
    ExamMeta,
    QuestionSlot,
    ReqVector,
    RevisionEntry,
    SectionStructure,
)
from config.defaults import (
    POINTS_PER_TYPE,
    PROMPTS_DIR,
    QUESTION_COUNT,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "Exam_Planner"
PROMPT_PATH = PROMPTS_DIR / "exam_planner.md"


# ────────────────────────────────────────────────────────────
# 결정론적 헬퍼 — Step 2 (그룹 → 레벨 세분화)
# ────────────────────────────────────────────────────────────
def split_group_to_levels(
    group_distribution: DifficultyDistributionGroup,
) -> dict[str, int]:
    """그룹 단위 분포를 5단계 레벨 분포로 세분화.

    규칙:
        easy (n)   → L1 ceil(n/2), L2 floor(n/2)
        medium (n) → L3 n
        hard (n)   → L4 ceil(n/2), L5 floor(n/2)

    >>> result = split_group_to_levels(DifficultyDistributionGroup(easy=2, medium=6, hard=2))
    >>> result
    {'L1': 1, 'L2': 1, 'L3': 6, 'L4': 1, 'L5': 1}
    """
    n_easy = group_distribution.easy
    n_med = group_distribution.medium
    n_hard = group_distribution.hard

    l1 = math.ceil(n_easy / 2)
    l2 = n_easy - l1
    l3 = n_med
    l4 = math.ceil(n_hard / 2)
    l5 = n_hard - l4

    return {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5}


# ────────────────────────────────────────────────────────────
# 결정론적 헬퍼 — Step 3 (섹션 구조)
# ────────────────────────────────────────────────────────────
def build_section_structure(
    req: ReqVector,
) -> list[SectionStructure]:
    """문항 유형 분포로 섹션 구조 구성.

    같은 유형은 한 섹션으로 묶음. POINTS_PER_TYPE을 기본 배점으로 사용.
    """
    sections: list[SectionStructure] = []
    section_idx = 1
    type_dist = req.question_type_distribution

    type_order = [
        QuestionType.SHORT_ANSWER,
        QuestionType.LONG_ANSWER,
        QuestionType.CASE_ANALYSIS,
        QuestionType.MCQ_SINGLE,
    ]
    type_labels = {
        QuestionType.SHORT_ANSWER: "단답형",
        QuestionType.LONG_ANSWER: "서술형",
        QuestionType.CASE_ANALYSIS: "응용/사례형",
        QuestionType.MCQ_SINGLE: "객관식",
    }

    for qtype in type_order:
        n = getattr(type_dist, qtype.value)
        if n <= 0:
            continue
        sections.append(SectionStructure(
            section_id=f"S{section_idx}",
            section_title=type_labels[qtype],
            question_type=qtype,
            num_questions=n,
            points_per_question=POINTS_PER_TYPE[qtype.value],
        ))
        section_idx += 1

    return sections


# ────────────────────────────────────────────────────────────
# 결정론적 헬퍼 — 슬롯 단위 배정 계획
# ────────────────────────────────────────────────────────────
def plan_slot_assignments(
    sections: list[SectionStructure],
    level_distribution: dict[str, int],
) -> list[tuple[str, str, QuestionType, DifficultyLevel]]:
    """섹션 × 레벨을 슬롯 단위로 펼침. (LLM 호출 전 사전 계획)

    제약을 만족하는 (section, slot_id, type, level) 튜플 리스트를 반환.

    알고리즘:
        1. 각 유형이 만들 수 있는 레벨 범위를 가져옴 (QUESTION_TYPE_LEVEL_RANGE)
        2. 그리디하게 매칭:
           - case_analysis 같은 제약이 강한 유형부터 먼저 배정
           - level별 남은 자리에 type별 남은 자리를 채움
        3. 모든 유형·레벨이 소진되어야 성공

    Returns:
        [(slot_id, section_id, question_type, level), ...]
    """
    # 레벨별 남은 수
    remaining_levels: dict[int, int] = {
        i: level_distribution.get(f"L{i}", 0) for i in range(1, 6)
    }

    # 섹션별 남은 슬롯 (유형 포함)
    remaining_sections: list[dict] = [
        {
            "section_id": s.section_id,
            "question_type": (
                QuestionType(s.question_type)
                if isinstance(s.question_type, str)
                else s.question_type
            ),
            "remaining": s.num_questions,
        }
        for s in sections
    ]

    # 그리디: 제약이 강한 유형부터 — case_analysis(L3~5), long(L2~5), short(L1~4)
    type_priority = {
        QuestionType.CASE_ANALYSIS: 0,
        QuestionType.LONG_ANSWER: 1,
        QuestionType.SHORT_ANSWER: 2,
        QuestionType.MCQ_SINGLE: 3,
    }
    remaining_sections.sort(
        key=lambda s: type_priority.get(s["question_type"], 99)
    )

    assignments: list[tuple[str, str, QuestionType, DifficultyLevel]] = []
    slot_counter = 1

    for section in remaining_sections:
        qtype = section["question_type"]
        # 이 유형이 만들 수 있는 레벨 (제약 적용)
        valid_levels = [
            DifficultyLevel(i) for i in range(1, 6)
            if is_valid_type_level_combo(qtype, DifficultyLevel(i))
        ]
        # 어려운 레벨부터 채움 (case_analysis가 L5를 먼저 가져가도록)
        valid_levels.sort(reverse=True)

        for _ in range(section["remaining"]):
            assigned_level: Optional[DifficultyLevel] = None
            for level in valid_levels:
                if remaining_levels[int(level)] > 0:
                    assigned_level = level
                    remaining_levels[int(level)] -= 1
                    break

            if assigned_level is None:
                raise ValueError(
                    f"유형-레벨 제약 만족 불가: section={section['section_id']}, "
                    f"type={qtype.value}, 남은 레벨={remaining_levels}"
                )

            assignments.append((
                f"Q{slot_counter:02d}",
                section["section_id"],
                qtype,
                assigned_level,
            ))
            slot_counter += 1

    # 모든 레벨 소진 확인
    leftover = {k: v for k, v in remaining_levels.items() if v > 0}
    if leftover:
        raise ValueError(f"미배정 레벨이 남음: {leftover}")

    return assignments


# ────────────────────────────────────────────────────────────
# LLM 호출 — concept 배정
# ────────────────────────────────────────────────────────────
def build_concept_assignment_prompt(
    assignments: list[tuple[str, str, QuestionType, DifficultyLevel]],
    structure: ConceptKnowledgeStructure,
    target_chapters: list[str],
) -> str:
    """LLM이 받을 user prompt 구성.

    각 슬롯에 어떤 concept을 배정할지 LLM에게 의미적 판단 요청.
    """
    assignable = get_assignable_nodes(structure)
    # 노드 정보를 간결하게 (embedding 제외)
    node_summaries = [
        {
            "concept_id": n.concept_id,
            "name": n.concept_name,
            "depth": n.depth_in_tree,
            "parent": n.parent_concept_id,
            "intrinsic_level": int(n.intrinsic_difficulty_level),
            "suitable_types": [t.value if hasattr(t, 'value') else t
                               for t in n.suitable_question_types],
            "importance": n.importance.value if hasattr(n.importance, 'value')
                          else n.importance,
        }
        for n in assignable
    ]

    slot_summaries = [
        {
            "slot_id": slot_id,
            "section_id": section_id,
            "question_type": qtype.value,
            "target_difficulty_level": int(level),
        }
        for slot_id, section_id, qtype, level in assignments
    ]

    user_prompt = f"""다음은 시험 출제 계획입니다.

## 시험 범위 (반드시 모두 커버해야 할 챕터/개념)
{json.dumps(target_chapters, ensure_ascii=False)}

## 사용 가능한 concept 노드 ({len(node_summaries)}개)
{json.dumps(node_summaries, ensure_ascii=False, indent=2)}

## 슬롯별 배정해야 할 정보 ({len(slot_summaries)}개)
{json.dumps(slot_summaries, ensure_ascii=False, indent=2)}

## 요청
각 슬롯에 `primary_concept_ids`와 `secondary_concept_ids`를 배정하세요.

규칙:
- target_difficulty_level이 가리키는 X₁ 값에 맞도록 노드 간 거리를 조정.
- L1 슬롯: primary_concept_ids 1개 (단일 노드)
- L2 슬롯: primary 1개, 인접 형제 노드 (parent 공유)
- L3 슬롯: primary 1~2개, 동일 부모 공유 복수 노드
- L4 슬롯: primary 2개, 서로 다른 범주(parent가 다른) 노드
- L5 슬롯: primary **3개 이상**, 광범위 모듈 통합
- concept의 suitable_types에 슬롯의 question_type이 포함되어야 함.
- target_chapters의 모든 항목이 어느 한 슬롯의 primary 또는 secondary에 등장.
- ROOT_DOCUMENT는 절대 사용하지 마세요.

## 출력 형식
다음 JSON 형식으로만 출력. 다른 텍스트 포함 금지.

```json
{{
  "slot_assignments": [
    {{
      "slot_id": "Q01",
      "primary_concept_ids": ["..."],
      "secondary_concept_ids": ["..."],
      "rationale": "이 배정의 근거 (1-2문장)"
    }}
  ],
  "coverage_check": {{
    "covered_chapters": ["..."],
    "missing_chapters": ["..."]
  }}
}}
```
"""
    return user_prompt


# ────────────────────────────────────────────────────────────
# 메인 Planner 클래스
# ────────────────────────────────────────────────────────────
class ExamPlanner:
    """Exam Planner Agent.

    Usage:
        planner = ExamPlanner(gemini_client=GeminiClient(mock=True))
        blueprint = planner.plan(req_vector, knowledge_structure, session_id)
    """

    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def plan(
        self,
        req: ReqVector,
        structure: ConceptKnowledgeStructure,
        session_id: str,
        *,
        previous_blueprint: Optional[ExamBlueprint] = None,
        correction_reason: Optional[str] = None,
    ) -> ExamBlueprint:
        """Blueprint 생성. previous_blueprint가 있으면 Difficulty Correction 모드.

        Args:
            req: 시험 요구사항
            structure: 지식 구조
            session_id: 세션 ID
            previous_blueprint: 이전 blueprint (correction 모드일 때만)
            correction_reason: correction 사유

        Returns:
            ExamBlueprint
        """
        is_correction = previous_blueprint is not None
        if is_correction:
            return self._correct(
                req, structure, session_id, previous_blueprint, correction_reason
            )
        return self._plan_initial(req, structure, session_id)

    # ────── 초기 계획 ──────
    def _plan_initial(
        self,
        req: ReqVector,
        structure: ConceptKnowledgeStructure,
        session_id: str,
    ) -> ExamBlueprint:
        warnings: list[str] = []

        # Step 1: 가중치
        focus = EvaluationFocus(req.evaluation_focus) if isinstance(
            req.evaluation_focus, str
        ) else req.evaluation_focus
        alpha, beta = get_weights(focus)

        # Step 2: 그룹 → 레벨
        group_dist = req.difficulty_distribution_group
        level_dist = split_group_to_levels(group_dist)

        # Step 3: 섹션 구조
        sections = build_section_structure(req)

        # 검증: section 합 == 레벨 합
        section_total = sum(s.num_questions for s in sections)
        level_total = sum(level_dist.values())
        if section_total != level_total:
            warnings.append(
                f"문항 수 불일치: section={section_total}, level={level_total}"
            )

        # Step 4: 슬롯 사전 배정 (LLM 호출 전 결정론적 계획)
        try:
            assignments = plan_slot_assignments(sections, level_dist)
        except ValueError as e:
            raise ValueError(
                f"슬롯 사전 배정 실패: {e}. "
                f"req_vector의 question_type_distribution과 "
                f"difficulty_distribution_group이 유형-레벨 제약을 만족하는지 확인."
            )

        # Step 5: LLM에 concept 배정 요청
        user_prompt = build_concept_assignment_prompt(
            assignments, structure, req.target_chapters_or_concepts
        )

        llm_result = self.client.generate_json(
            agent_name=AGENT_NAME,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )

        # Step 6: LLM 응답을 QuestionSlot으로 변환 + X1/X2/D 계산
        slot_assignments = llm_result.get("slot_assignments", [])
        assignment_by_id = {
            a["slot_id"]: a for a in slot_assignments
        }

        slots: list[QuestionSlot] = []
        for slot_id, section_id, qtype, level in assignments:
            llm_assignment = assignment_by_id.get(slot_id)
            if not llm_assignment:
                warnings.append(f"LLM 응답에 {slot_id} 누락 — skip")
                continue

            primary_ids = llm_assignment.get("primary_concept_ids", [])
            secondary_ids = llm_assignment.get("secondary_concept_ids", [])

            # ROOT 필터링
            primary_ids = [
                cid for cid in primary_ids if cid != ROOT_CONCEPT_ID
            ]
            secondary_ids = [
                cid for cid in secondary_ids if cid != ROOT_CONCEPT_ID
            ]

            if not primary_ids:
                warnings.append(
                    f"{slot_id}: primary_concept_ids 비어있음 — skip"
                )
                continue

            # X1: 배정된 primary concept 간 그래프 최대 hop 거리를 0~1로 정규화
            #   - 단일 개념(거리=0) → X1=0.00 (L1 범위)
            #   - 인접 형제 개념(거리=1) → X1=0.25
            #   - 여러 범주 교차(거리≥4) → X1=1.00 (L5 범위)
            max_dist = get_max_distance_in_subset(structure, primary_ids)
            x1 = compute_x1(max(0, max_dist))

            # X2: 문항 유형에 따른 인지 부하 (Bloom's Taxonomy 기반)
            #   - short_answer=0.4, long_answer=0.7, case_analysis=1.0
            x2 = compute_x2(qtype)

            # D = α·X1 + β·X2  (BALANCED: α=β=0.5)
            # D 값을 L1~L5 정수 레벨로 매핑
            d = compute_d(x1, x2, focus)
            computed_level = d_to_level(d)

            # 검증: 계산된 레벨과 목표 레벨이 어긋나면 warning (코드는 진행)
            if computed_level != level:
                warnings.append(
                    f"{slot_id}: target=L{int(level)}, computed=L{int(computed_level)} "
                    f"(D={d:.3f}, X1={x1}, X2={x2}). 슬롯의 concept 배정이 부정확할 수 있음."
                )

            # points: section의 기본값 사용
            section = next(s for s in sections if s.section_id == section_id)

            slots.append(QuestionSlot(
                slot_id=slot_id,
                section_id=section_id,
                question_type=qtype,
                target_difficulty_level=level,
                expected_X1=x1,
                expected_X2=x2,
                expected_D=round(d, 4),
                points=section.points_per_question,
                primary_concept_ids=primary_ids,
                secondary_concept_ids=secondary_ids,
                special_instructions=None,
            ))

        # 커버리지 체크
        coverage = llm_result.get("coverage_check", {})
        missing = coverage.get("missing_chapters", [])
        if missing:
            warnings.append(f"미커버 챕터: {missing}")

        # 배점 합 조정: 문항 유형별 단가(POINTS_PER_TYPE)의 합이 total_points와
        # 정확히 일치하지 않을 수 있음 (예: 2×10 + 5×12 + 3×7 = 101 ≠ 100).
        # 초과분은 제약이 가장 느슨한 case_analysis 슬롯 뒤에서부터 1점씩 차감.
        total_pts_computed = sum(s.points for s in slots)
        excess = total_pts_computed - req.total_points
        if excess != 0:
            ca_indices = [
                i for i, s in enumerate(slots)
                if s.question_type == QuestionType.CASE_ANALYSIS
            ]
            if 0 < excess <= len(ca_indices):
                for idx in ca_indices[-excess:]:
                    slots[idx].points -= 1
            else:
                warnings.append(
                    f"배점 합 불일치: computed={total_pts_computed}, "
                    f"req={req.total_points} — 자동 조정 불가"
                )

        # Blueprint 조립
        blueprint = ExamBlueprint(
            session_id=session_id,
            exam_meta=ExamMeta(
                exam_title=req.exam_title,
                exam_type=req.exam_type,
                total_points=req.total_points,
                duration_minutes=req.duration_minutes,
                target_chapters_or_concepts=req.target_chapters_or_concepts,
                evaluation_focus=focus,
                weights_applied={"alpha": alpha, "beta": beta},
                allowed_formats=req.allowed_formats,
            ),
            difficulty_policy=DifficultyPolicy(
                baseline_level=DifficultyLevel.L3,
                target_distribution_group=group_dist,
                target_distribution_level=level_dist,
            ),
            section_structure=sections,
            question_slots=slots,
            revision_history=[
                RevisionEntry(
                    revision_id=0,
                    trigger="initial",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    changed_slots=[s.slot_id for s in slots],
                    reason="initial generation",
                )
            ],
            warnings=warnings,
        )
        return blueprint

    # ────── Difficulty Correction ──────
    def _correct(
        self,
        req: ReqVector,
        structure: ConceptKnowledgeStructure,
        session_id: str,
        previous: ExamBlueprint,
        reason: Optional[str],
    ) -> ExamBlueprint:
        """현재는 stub — 후속 슬롯 레벨 조정 로직.

        실 운영에선 Difficulty_Tester가 어느 슬롯이 문제였는지 함께 전달해야 함.
        지금은 v0.1로, 단순히 revision_history만 추가하고 그대로 반환.
        """
        revised = previous.model_copy(deep=True)
        revised.revision_history.append(
            RevisionEntry(
                revision_id=len(previous.revision_history),
                trigger="difficulty_correction",
                timestamp=datetime.now(timezone.utc).isoformat(),
                changed_slots=[],
                reason=reason or "difficulty correction triggered",
            )
        )
        revised.warnings.append(
            "Difficulty Correction stub: 실제 슬롯 조정 로직 미구현 (v0.1)"
        )
        return revised
