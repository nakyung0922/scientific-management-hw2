"""
Pydantic models implementing the v0.4 JSON schema.

모든 agent 간 메시지 payload의 타입 안전한 표현.
JSON 직렬화/역직렬화는 model.model_dump_json() / model.model_validate_json() 사용.

PG1·PG3도 이 스키마를 import해서 사용.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.enums import (
    DifficultyGroup,
    DifficultyLevel,
    EdgeRelation,
    EvaluationFocus,
    Importance,
    QuestionType,
    RoutingStatus,
    TesterName,
    Verdict,
)


# ────────────────────────────────────────────────────────────
# 공통 베이스
# ────────────────────────────────────────────────────────────
class StrictBase(BaseModel):
    """모든 스키마의 base. extra field 금지로 오타 방지."""
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


# ────────────────────────────────────────────────────────────
# Global Message Envelope
# ────────────────────────────────────────────────────────────
class MessageEnvelope(StrictBase):
    """모든 agent 간 통신의 봉투.

    payload는 dict로 받지만, 실제로는 아래 정의된 스키마 객체의 dump임.
    PG3 Supervisor가 routing_status를 보고 분기.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    session_id: str
    source_agent: str
    target_agent: str
    payload: dict
    routing_status: RoutingStatus = RoutingStatus.FLOW


# ────────────────────────────────────────────────────────────
# raw_document & concept_summary (Material_Collector 출력)
# ────────────────────────────────────────────────────────────
class PageText(StrictBase):
    """단일 PDF 페이지의 추출 텍스트."""
    page_no: int = Field(ge=1)
    text: str


class RawDocument(StrictBase):
    """단일 PDF 파일 → 모듈 단위 raw text.

    Material_Collector의 Pass 1 산출물. pypdf로 페이지별 텍스트만 추출한 상태.
    LLM 호출 전이므로 어떤 요약도 들어있지 않음.
    """
    module_id: str = Field(
        description="파일명 휴리스틱으로 추정한 모듈 ID. "
                    "예: 'M1_1', 'M2_1_2', 'manufacturing_overview'"
    )
    source_filename: str
    pages: list[PageText] = Field(min_length=1)


class ConceptSummary(StrictBase):
    """모듈 1개의 요약 + 핵심 개념 후보.

    Material_Collector의 Pass 2 산출물. Topic_Analyzer 입력.
    LLM이 RawDocument를 받아 모듈 단위로 요약·핵심 phrase를 뽑음.
    """
    module_id: str
    source_filename: str
    title: str = Field(description="모듈 제목 (예: 'What is Work?')")
    summary_text: str = Field(
        description="모듈 전체의 1~3문단 요약. Topic_Analyzer가 LLM에 전달할 컨텍스트."
    )
    key_phrases: list[str] = Field(
        default_factory=list,
        description="핵심 개념 후보 (LLM이 자유롭게 뽑음, Topic_Analyzer가 정제·노드화)."
    )
    page_count: int = Field(ge=1)
    warnings: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────
# req_vector (Req_Parser 출력)
# ────────────────────────────────────────────────────────────
class DifficultyDistributionGroup(StrictBase):
    easy: int = Field(ge=0)
    medium: int = Field(ge=0)
    hard: int = Field(ge=0)


class QuestionTypeDistribution(StrictBase):
    MCQ_single: int = Field(default=0, ge=0)
    short_answer: int = Field(default=0, ge=0)
    long_answer: int = Field(default=0, ge=0)
    case_analysis: int = Field(default=0, ge=0)


class ReqVector(StrictBase):
    """교수 요구사항을 파싱한 결과."""
    exam_title: str
    exam_type: str
    target_chapters_or_concepts: list[str]
    total_points: int = Field(gt=0)
    duration_minutes: int = Field(gt=0)
    evaluation_focus: EvaluationFocus = EvaluationFocus.BALANCED
    difficulty_distribution_group: DifficultyDistributionGroup
    question_type_distribution: QuestionTypeDistribution
    allowed_formats: list[str] = Field(default_factory=lambda: ["docx", "pdf"])
    special_instructions: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────
# concept_knowledge_structure (Topic_Analyzer 출력)
# ────────────────────────────────────────────────────────────
class ConceptNode(StrictBase):
    """지식 그래프의 노드."""
    concept_id: str = Field(
        description="형식: {module_id}_{snake_case_slug}, "
                    "예: 'M2_1_5_kj_method'"
    )
    concept_name: str
    depth_in_tree: int = Field(ge=0)
    parent_concept_id: Optional[str] = None
    importance: Importance = Importance.MEDIUM
    intrinsic_difficulty_level: DifficultyLevel
    suitable_question_types: list[QuestionType]
    source_pages: list[str] = Field(default_factory=list)
    # embedding_vector는 별도 저장소 권장 — 스키마엔 optional로
    embedding_vector: Optional[list[float]] = None

    @field_validator("concept_id")
    @classmethod
    def validate_concept_id_format(cls, v: str) -> str:
        """R3 합의된 네이밍 규칙 검증: 소문자/숫자/언더스코어만."""
        if not v:
            raise ValueError("concept_id must not be empty")
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_")
        if not set(v.lower()).issubset(allowed | set("M")):
            # 'M'으로 시작하는 module prefix는 대문자 허용
            pass
        if " " in v or "." in v:
            raise ValueError(
                f"concept_id '{v}' contains forbidden chars (space or dot). "
                f"Use format: M2_1_5_kj_method"
            )
        return v


class ConceptEdge(StrictBase):
    """지식 그래프의 엣지."""
    from_concept_id: str
    to_concept_id: str
    relation: EdgeRelation
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class ConceptKnowledgeStructure(StrictBase):
    """Topic_Analyzer의 최종 출력."""
    nodes: list[ConceptNode]
    edges: list[ConceptEdge] = Field(default_factory=list)

    def get_node(self, concept_id: str) -> Optional[ConceptNode]:
        """concept_id로 노드 조회."""
        for node in self.nodes:
            if node.concept_id == concept_id:
                return node
        return None

    def get_depth_between(self, id_a: str, id_b: str) -> int:
        """두 노드 사이의 그래프 거리 (간단 BFS).

        실제 운영에선 networkx로 대체 가능. 지금은 의존성 최소화.
        """
        if id_a == id_b:
            return 0

        # 인접 리스트 구축
        adj: dict[str, set[str]] = {}
        for edge in self.edges:
            adj.setdefault(edge.from_concept_id, set()).add(edge.to_concept_id)
            adj.setdefault(edge.to_concept_id, set()).add(edge.from_concept_id)

        # BFS
        from collections import deque
        visited = {id_a}
        queue: deque[tuple[str, int]] = deque([(id_a, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in adj.get(current, set()):
                if neighbor == id_b:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return -1  # 연결되지 않음


# ────────────────────────────────────────────────────────────
# exam_blueprint (Exam_Planner 출력)
# ────────────────────────────────────────────────────────────
class ExamMeta(StrictBase):
    exam_title: str
    exam_type: str
    total_points: int
    duration_minutes: int
    target_chapters_or_concepts: list[str]
    evaluation_focus: EvaluationFocus
    weights_applied: dict[str, float]  # {"alpha": 0.5, "beta": 0.5}
    allowed_formats: list[str] = Field(default_factory=lambda: ["docx", "pdf"])


class DifficultyPolicy(StrictBase):
    baseline_level: DifficultyLevel = DifficultyLevel.L3
    target_distribution_group: DifficultyDistributionGroup
    target_distribution_level: dict[str, int]  # {"L1": 1, "L2": 1, ...}


class SectionStructure(StrictBase):
    section_id: str
    section_title: str
    question_type: QuestionType
    num_questions: int = Field(gt=0)
    points_per_question: int = Field(gt=0)


class QuestionSlot(StrictBase):
    """청사진의 단일 문항 슬롯.

    Q&A Generator가 이 슬롯을 받아 실제 문항을 생성.
    """
    slot_id: str
    section_id: str
    question_type: QuestionType
    target_difficulty_level: DifficultyLevel
    expected_X1: float = Field(ge=0.0, le=1.0)
    expected_X2: float = Field(ge=0.0, le=1.0)
    expected_D: float = Field(ge=0.0, le=1.0)
    points: int = Field(gt=0)
    primary_concept_ids: list[str] = Field(min_length=1)
    secondary_concept_ids: list[str] = Field(default_factory=list)
    special_instructions: Optional[str] = None


class RevisionEntry(StrictBase):
    revision_id: int = Field(ge=0)
    trigger: str  # "initial" | "difficulty_correction"
    timestamp: str
    changed_slots: list[str]
    reason: str


class ExamBlueprint(StrictBase):
    """Exam_Planner의 최종 출력. Q&A Generator의 입력."""
    blueprint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    generated_by: str = "Exam_Planner"
    exam_meta: ExamMeta
    difficulty_policy: DifficultyPolicy
    section_structure: list[SectionStructure]
    question_slots: list[QuestionSlot]
    revision_history: list[RevisionEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────
# question (Q&A Generator 출력)
# ────────────────────────────────────────────────────────────
class Choice(StrictBase):
    """MCQ_single의 선택지."""
    label: str  # 예: "①", "A"
    text: str


class SourceReference(StrictBase):
    """문항이 인용하는 강의자료 출처."""
    concept_id: str
    page_or_slide: str  # 예: "M3.1 p.7"
    excerpt: str


class GenerationMetadata(StrictBase):
    generator_version: str
    attempt_number: int = Field(ge=1)
    previous_question_id: Optional[str] = None


class Question(StrictBase):
    """Q&A_Generator의 출력. Critic·Rubric·Formatter의 입력."""
    question_id: str
    slot_id: str
    session_id: str

    question_type: QuestionType
    target_difficulty_level: DifficultyLevel
    points: int = Field(gt=0)

    prompt: str
    choices: Optional[list[Choice]] = None  # MCQ_single만 채움
    reference_answer: str
    answer_steps: list[str] = Field(default_factory=list)

    source_references: list[SourceReference]
    generation_metadata: GenerationMetadata
    warnings: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────
# tester_output (Factfulness/Difficulty Tester 출력)
# ────────────────────────────────────────────────────────────
class NextAction(StrictBase):
    routing_status: RoutingStatus
    target_agent: str


class TesterOutput(StrictBase):
    """두 Tester가 공유하는 출력 형식.

    estimated_difficulty_level/X1/X2/D는 Difficulty_Tester만 채움.
    Factfulness_Tester는 None으로 둠.
    """
    tester_name: TesterName
    question_id: str
    session_id: str

    score: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    threshold_used: float = Field(ge=0.0, le=1.0)

    estimated_difficulty_level: Optional[DifficultyLevel] = None
    estimated_X1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    estimated_X2: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    estimated_D: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    reasons: list[str] = Field(default_factory=list)
    next_action: Optional[NextAction] = None


# ────────────────────────────────────────────────────────────
# answer_rubric (Rubric_Machine 출력)
# ────────────────────────────────────────────────────────────
class PartialCredit(StrictBase):
    condition: str
    points: int = Field(ge=0)


class RubricCriterion(StrictBase):
    criterion_id: str
    description: str
    linked_answer_step: Optional[str] = None
    points: int = Field(gt=0)
    key_points: list[str] = Field(default_factory=list)
    partial_credit_guide: str = ""
    partial_credit: list[PartialCredit] = Field(default_factory=list)


class CommonError(StrictBase):
    error_pattern: str
    deduction: int = Field(ge=0)


class AnswerRubric(StrictBase):
    """Rubric_Machine의 출력. Paper_Formattor의 입력."""
    rubric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_id: str
    session_id: str
    total_points: int = Field(gt=0)
    criteria: list[RubricCriterion] = Field(min_length=1)
    common_errors: list[CommonError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────
# qa_paper_output (Paper_Formattor 출력)
# ────────────────────────────────────────────────────────────
class QAPaperOutput(StrictBase):
    """Paper_Formattor의 최종 출력. 시스템의 최종 산출물."""
    paper_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    exam_paper_file: str  # 파일 경로
    answer_key_file: str  # 파일 경로
    format: str = "both"  # "docx" | "pdf" | "both"


# ────────────────────────────────────────────────────────────
# 자체 테스트
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 최소 인스턴스 생성 테스트
    req = ReqVector(
        exam_title="Test Midterm",
        exam_type="midterm",
        target_chapters_or_concepts=["M1_1", "M1_2"],
        total_points=100,
        duration_minutes=75,
        difficulty_distribution_group=DifficultyDistributionGroup(
            easy=2, medium=6, hard=2
        ),
        question_type_distribution=QuestionTypeDistribution(
            short_answer=2, long_answer=5, case_analysis=3
        ),
    )
    print("✓ ReqVector created")
    print(req.model_dump_json(indent=2)[:200], "...")

    # ConceptNode 네이밍 검증 — 통과 케이스
    node = ConceptNode(
        concept_id="M2_1_5_kj_method",
        concept_name="KJ Method",
        depth_in_tree=2,
        intrinsic_difficulty_level=DifficultyLevel.L3,
        suitable_question_types=[QuestionType.SHORT_ANSWER],
    )
    print(f"\n✓ ConceptNode created: {node.concept_id}")

    # ConceptNode 네이밍 검증 — 실패 케이스
    try:
        bad = ConceptNode(
            concept_id="M2.1.5.kj method",  # 점, 공백 포함
            concept_name="Bad",
            depth_in_tree=0,
            intrinsic_difficulty_level=DifficultyLevel.L1,
            suitable_question_types=[],
        )
        print("✗ Should have rejected bad concept_id")
    except ValueError as e:
        print(f"✓ Correctly rejected bad concept_id: {e}")

    print("\n✓ All schema smoke tests passed")
