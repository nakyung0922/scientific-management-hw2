"""
Pydantic schemas for Material Collector output.

설계 원칙
----------
- PG2 `common.schemas.StrictBase` 상속 → extra="forbid"로 타입 안전.
- Material Collector 의 output payload 는 Topic_Analyzer 의 input 이 됨.
- Topic_Analyzer 가 `ConceptNode(concept_id=...)` 를 생성할 때 필요한 모든
  재료를 여기서 미리 준비:
    * concept_id: v0.5 네이밍 규칙 {module_id}_{snake_case_slug} 따름
    * source_pages: ConceptNode.source_pages 와 1:1 대응
    * raw_text: RAG 근거 인용 / Hallucination 검증용

본 스키마는 ConceptKnowledgeStructure(노드/엣지) 의 "원재료" 단계이며,
실제 그래프 구축(엣지 추론, depth_in_tree, intrinsic_difficulty_level 산출)은
다음 에이전트 Topic_Analyzer 책임.
"""
from __future__ import annotations

from typing import ClassVar, Optional

from pydantic import Field, field_validator

from common.schemas import StrictBase


class ConceptUnit(StrictBase):
    """강의자료에서 추출한 개념 단위 (Topic_Analyzer 입력 재료).

    Topic_Analyzer 가 이것을 바탕으로 ConceptNode 를 생성한다.
    concept_id 는 v0.5 네이밍 규칙을 따라 Material Collector 가 미리 만들어둠.
    """

    concept_id: str = Field(
        description=(
            "v0.5 네이밍 규칙: {module_id}_{snake_case_slug}. "
            "예: 'M1_4_taylor_principles'. "
            "module_id 는 TARGET_CHAPTERS 의 prefix 또는 파일명에서 추론."
        )
    )
    concept_name: str = Field(
        description="개념의 한 줄 이름 (60자 이내). LLM 이 생성."
    )
    summary: str = Field(
        description="개념 요약 2~5 문장. 임베딩 대상 텍스트."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="3~7 개의 핵심 키워드. 메타 필터/가중치용.",
    )

    # ── 출처 추적 (Hallucination 방지) ────────────────────────────
    source_file: str = Field(
        description="원본 PDF 파일명 (basename)."
    )
    source_pages: list[str] = Field(
        description=(
            "이 개념을 추출한 페이지들. v0.5 ConceptNode.source_pages 형식을 따름. "
            "예: ['M5.1 p.10', 'M5.1 p.11']."
        )
    )
    raw_text: str = Field(
        description="원문 텍스트 (RAG 근거 인용용). 비어있을 수 있음."
    )

    # ── 메타 ──────────────────────────────────────────────────────
    section_title: Optional[str] = Field(
        default=None,
        description="추정된 슬라이드/섹션 제목. 없을 수 있음.",
    )
    extraction_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "추출 신뢰도. OCR 페이지는 0.7, 텍스트 페이지는 1.0, "
            "LLM 응답 fallback 은 0.3 등."
        ),
    )

    # ── concept_id 형식 검증 (v0.5 규칙) ───────────────────────────
    _ALLOWED_CHARS: ClassVar[set[str]] = set(
        "abcdefghijklmnopqrstuvwxyz0123456789_M"
    )

    @field_validator("concept_id")
    @classmethod
    def validate_concept_id(cls, v: str) -> str:
        if not v:
            raise ValueError("concept_id 가 비어있음")
        if " " in v or "." in v:
            raise ValueError(
                f"concept_id '{v}' 에 공백/점 포함 — 언더스코어로 변환 필요"
            )
        # 점/공백 제외 추가 검증
        bad = set(v) - cls._ALLOWED_CHARS
        if bad:
            raise ValueError(
                f"concept_id '{v}' 에 허용되지 않는 문자 포함: {sorted(bad)}"
            )
        return v


class MaterialCollectorOutput(StrictBase):
    """Material Collector 의 최종 출력 payload.

    MessageEnvelope.payload 에 들어가는 본문.
    Topic_Analyzer 가 받음.
    """

    session_id: str
    generated_by: str = "Material_Collector"

    course_name: str
    source_files: list[str] = Field(
        description="처리한 PDF 파일 목록 (basename)."
    )
    total_pages: int = Field(ge=0)
    total_concepts: int = Field(ge=0)

    concepts: list[ConceptUnit] = Field(
        description="추출된 개념 단위 리스트. Topic_Analyzer 가 노드로 변환."
    )

    # ── 라벨링 결과 ────────────────────────────────────────────────
    # Material Collector 는 시험 범위에 가정하지 않고, 들어온 파일마다 안정적인
    # module_id 라벨을 부여한다. 이 매핑이 Topic_Analyzer / Exam_Planner 가
    # "어떤 PDF 가 어떤 module 로 라벨링됐는지" 추적하는 진실의 원천.
    file_to_module_id: dict[str, str] = Field(
        default_factory=dict,
        description="basename → module_id. 예: 'M1.4.pdf' → 'M1_4'.",
    )

    # ── 처리 메타 ─────────────────────────────────────────────────
    extraction_methods: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "추출 방법별 페이지 수. 예: {'text': 30, 'ocr': 6}. "
            "다음 에이전트가 OCR 비중 보고 신뢰도 가중 가능."
        ),
    )
    ocr_languages: list[str] = Field(
        default_factory=list,
        description="OCR 사용 시 언어 코드. 예: ['kor', 'eng'].",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="처리 중 경고 (스킵된 파일, LLM 실패, 신뢰도 낮은 청크 등).",
    )
