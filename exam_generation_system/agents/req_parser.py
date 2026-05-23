"""
Req Parser — Agent 1-2 (PG1, R3).

입력: 교수의 시험 요구사항 (dict | str | Path | None)
출력: ReqVector (Exam_Planner 입력)

설계 원칙:
- 어떤 입력이 와도 LLM 1회 호출로 정규화. dict로 들어와도 LLM이 보강.
- LLM 출력의 null 필드는 config/defaults.py 기본값으로 채움.
- 정합성 깨진 경우 (합계 불일치, enum 위반 등) → defaults로 fallback + warnings 기록.
- 시스템을 절대 멈추지 않음. 무엇이 들어와도 valid ReqVector를 반환.

LOA = 8 (사람 개입 거의 없음, 거의 결정 자동화).
"""
from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import Field, ValidationError

from common.enums import EvaluationFocus
from common.gemini_client import GeminiClient
from common.schemas import (
    DifficultyDistributionGroup,
    QuestionTypeDistribution,
    ReqVector,
    StrictBase,
)
from config.defaults import (
    DIFFICULTY_DISTRIBUTION_GROUP,
    EXAM_BASICS,
    EXAM_HEADER_TEMPLATE,
    EXAM_METADATA,
    PROJECT_ROOT,
    QUESTION_COUNT,
    TARGET_CHAPTERS,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "Req_Parser"
PROMPT_REL_PATH = "config/prompts/req_parser.md"


# ────────────────────────────────────────────────────────────
# LLM raw output 검증 모델 (모든 필드 Optional)
# ────────────────────────────────────────────────────────────
class ParsedRawOutput(StrictBase):
    """LLM이 반환하는 raw JSON. 모든 필드가 null 가능."""
    exam_title: Optional[str] = None
    exam_type: Optional[str] = None
    target_chapters_or_concepts: Optional[list[str]] = None
    total_points: Optional[int] = None
    duration_minutes: Optional[int] = None
    evaluation_focus: Optional[str] = None
    difficulty_distribution_group: Optional[dict] = None
    question_type_distribution: Optional[dict] = None
    allowed_formats: Optional[list[str]] = None
    special_instructions: Optional[str] = None
    # LLM이 발견한 모호함을 기록
    warnings_field: list[str] = Field(default_factory=list, alias="_warnings")


# ────────────────────────────────────────────────────────────
# 프롬프트 캐시
# ────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _load_prompt_file(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


# ────────────────────────────────────────────────────────────
# target 정규화 (v0.5 네이밍 규칙)
# ────────────────────────────────────────────────────────────
_MODULE_OR_CONCEPT_RE = re.compile(
    r"^M\d+(?:_\d+){1,3}(?:_[a-z0-9_]+)?$"  # M1_1 또는 M1_1_kj_method 등
)


def normalize_target(raw: str) -> Optional[str]:
    """단일 target 문자열을 v0.5 규칙에 맞게 정규화.

    >>> normalize_target("M1_1")
    'M1_1'
    >>> normalize_target("m1.1")
    'M1_1'
    >>> normalize_target("M1-1")
    'M1_1'
    >>> normalize_target("M2_1_2_KJ_Method")
    'M2_1_2_kj_method'
    >>> normalize_target("manufacturing_overview")
    'manufacturing_overview'
    >>> normalize_target("garbage 한글 ?")  # 매칭 실패
    """
    if not raw or not raw.strip():
        return None

    s = raw.strip()
    # 점·공백·하이픈을 언더스코어로
    s = re.sub(r"[.\s\-]+", "_", s)
    # 중복 언더스코어 정리
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return None

    # M prefix 대문자 + 슬러그는 소문자
    # "M1_1_KJ_Method" → "M1_1_kj_method"
    parts = s.split("_")
    if parts and parts[0].upper().startswith("M") and parts[0][1:].isdigit():
        head_parts = [parts[0].upper()]  # M1
        # 그 뒤에 오는 숫자들도 보존 (M1_1, M1_1_2)
        i = 1
        while i < len(parts) and parts[i].isdigit():
            head_parts.append(parts[i])
            i += 1
        tail_parts = [p.lower() for p in parts[i:]]
        normalized = "_".join(head_parts + tail_parts)
    else:
        # M으로 시작 안 함 → 그대로 소문자 (예: manufacturing_overview)
        normalized = s.lower()

    if _MODULE_OR_CONCEPT_RE.match(normalized) or re.match(r"^[a-z][a-z0-9_]*$", normalized):
        return normalized
    return None  # 정규화 실패


def normalize_target_list(raw_list: list[str]) -> tuple[list[str], list[str]]:
    """target 리스트 정규화. 중복 제거.

    Returns:
        (normalized_list, rejected_list) — rejected는 정규화 실패한 원본
    """
    seen: set[str] = set()
    normalized: list[str] = []
    rejected: list[str] = []

    for raw in raw_list:
        n = normalize_target(raw)
        if n is None:
            rejected.append(raw)
        elif n not in seen:
            seen.add(n)
            normalized.append(n)
    return normalized, rejected


# ────────────────────────────────────────────────────────────
# 입력 → 텍스트 변환
# ────────────────────────────────────────────────────────────
def _input_to_text(prof_input: Union[dict, str, Path, None]) -> str:
    """모든 입력 타입을 LLM에 넘길 텍스트로 변환."""
    if prof_input is None:
        return ""
    if isinstance(prof_input, dict):
        return json.dumps(prof_input, ensure_ascii=False, indent=2)
    if isinstance(prof_input, Path):
        if not prof_input.exists():
            raise FileNotFoundError(f"요구사항 파일 없음: {prof_input}")
        content = prof_input.read_text(encoding="utf-8")
        # .json 파일이면 dict로 처리한 것처럼
        if prof_input.suffix.lower() == ".json":
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                return content  # 깨졌으면 raw 텍스트로 LLM에 넘김
        return content
    if isinstance(prof_input, str):
        return prof_input
    raise TypeError(
        f"지원하지 않는 입력 타입: {type(prof_input)}. "
        f"dict | str | Path | None 중 하나여야 함."
    )


# ────────────────────────────────────────────────────────────
# Defaults 조립
# ────────────────────────────────────────────────────────────
def _default_exam_title() -> str:
    """defaults에서 시험 제목 자동 생성."""
    return (
        f"{EXAM_METADATA['semester']} "
        f"{EXAM_METADATA['course_name_en']} "
        f"{EXAM_HEADER_TEMPLATE['exam_label']}"
    )


def _default_question_type_distribution() -> dict:
    """defaults의 QUESTION_COUNT에서 dict 생성 (total 제외)."""
    return {
        "MCQ_single": 0,
        "short_answer": QUESTION_COUNT["short_answer"],
        "long_answer": QUESTION_COUNT["long_answer"],
        "case_analysis": QUESTION_COUNT["case_analysis"],
    }


# ────────────────────────────────────────────────────────────
# 메인 Parser 클래스
# ────────────────────────────────────────────────────────────
class ReqParser:
    """Req Parser Agent.

    Usage:
        parser = ReqParser(GeminiClient(...))
        req_vec = parser.parse(prof_input)
        # prof_input은 dict | str | Path | None 무엇이든 가능
    """

    def __init__(self, client: GeminiClient) -> None:
        self.client = client
        self.system_prompt = _load_prompt_file(str(PROJECT_ROOT / PROMPT_REL_PATH))

    # ── 공개 인터페이스 ──────────────────────────────────────
    def parse(self, prof_input: Union[dict, str, Path, None]) -> ReqVector:
        """모든 입력을 LLM을 통과시켜 정규화된 ReqVector 반환.

        실패해도 절대 raise하지 않음. defaults로 fallback.
        """
        warnings: list[str] = []

        # Step 1: 입력 텍스트화
        try:
            input_text = _input_to_text(prof_input)
        except (FileNotFoundError, TypeError) as exc:
            logger.warning("[Req_Parser] 입력 변환 실패: %s", exc)
            warnings.append(f"입력 처리 실패: {exc}. 전체 defaults 사용.")
            return self._build_full_defaults(warnings)

        # Step 2: LLM 호출 (빈 입력이어도 호출 — 일관성 위해)
        parsed: Optional[ParsedRawOutput] = None
        try:
            raw_json = self.client.generate_json(
                agent_name=AGENT_NAME,
                system_prompt=self.system_prompt,
                user_prompt=input_text or "(empty input)",
            )
            parsed = ParsedRawOutput.model_validate(raw_json)
        except (ValidationError, RuntimeError, ValueError) as exc:
            logger.warning("[Req_Parser] LLM 파싱 실패: %s", exc)
            warnings.append(f"LLM 파싱 실패: {exc}. 전체 defaults 사용.")
            return self._build_full_defaults(warnings)

        # Step 3: LLM이 발견한 warnings 흡수
        warnings.extend(parsed.warnings_field)

        # Step 4: 필드별로 defaults 보강 + 검증
        return self._build_from_parsed(parsed, warnings)

    # ── 전체 defaults 사용 ───────────────────────────────────
    def _build_full_defaults(self, warnings: list[str]) -> ReqVector:
        warnings.append("모든 필드 defaults 사용 (입력 부재 또는 처리 실패)")
        return ReqVector(
            exam_title=_default_exam_title(),
            exam_type=EXAM_BASICS["exam_type"],
            target_chapters_or_concepts=list(TARGET_CHAPTERS),
            total_points=EXAM_BASICS["total_points"],
            duration_minutes=EXAM_BASICS["duration_minutes"],
            evaluation_focus=EvaluationFocus(EXAM_BASICS["evaluation_focus"]),
            difficulty_distribution_group=DifficultyDistributionGroup(
                **DIFFICULTY_DISTRIBUTION_GROUP
            ),
            question_type_distribution=QuestionTypeDistribution(
                **_default_question_type_distribution()
            ),
            allowed_formats=["docx", "pdf"],
            special_instructions=None,
            warnings=warnings,
        )

    # ── LLM 파싱 결과 + defaults 조립 ────────────────────────
    def _build_from_parsed(
        self,
        parsed: ParsedRawOutput,
        warnings: list[str],
    ) -> ReqVector:
        # exam_title
        exam_title = parsed.exam_title or _default_exam_title()
        if not parsed.exam_title:
            warnings.append("exam_title: defaults 사용")

        # exam_type
        exam_type = parsed.exam_type or EXAM_BASICS["exam_type"]
        if not parsed.exam_type:
            warnings.append(f"exam_type: defaults '{exam_type}' 사용")

        # target_chapters_or_concepts (옵션 B: 모듈/concept 혼재 허용)
        target_chapters = self._resolve_targets(parsed, warnings)

        # total_points
        total_points = self._resolve_positive_int(
            parsed.total_points, EXAM_BASICS["total_points"],
            "total_points", warnings,
        )

        # duration_minutes
        duration_minutes = self._resolve_positive_int(
            parsed.duration_minutes, EXAM_BASICS["duration_minutes"],
            "duration_minutes", warnings,
        )

        # evaluation_focus (enum 검증)
        evaluation_focus = self._resolve_evaluation_focus(parsed, warnings)

        # difficulty_distribution_group (객체 통째로)
        difficulty_group = self._resolve_difficulty_group(parsed, warnings)

        # question_type_distribution (객체 통째로)
        type_dist = self._resolve_type_distribution(parsed, warnings)

        # 합계 일관성 검증
        self._check_distribution_consistency(difficulty_group, type_dist, warnings)

        # allowed_formats
        allowed_formats = parsed.allowed_formats or ["docx", "pdf"]

        # special_instructions
        special = parsed.special_instructions

        # 최종 ReqVector 조립
        try:
            return ReqVector(
                exam_title=exam_title,
                exam_type=exam_type,
                target_chapters_or_concepts=target_chapters,
                total_points=total_points,
                duration_minutes=duration_minutes,
                evaluation_focus=evaluation_focus,
                difficulty_distribution_group=difficulty_group,
                question_type_distribution=type_dist,
                allowed_formats=allowed_formats,
                special_instructions=special,
                warnings=warnings,
            )
        except ValidationError as exc:
            logger.warning(
                "[Req_Parser] 최종 ReqVector 검증 실패: %s. 전체 defaults 사용.",
                exc,
            )
            return self._build_full_defaults(
                [f"최종 검증 실패: {exc}"] + warnings
            )

    # ── 필드별 resolver ──────────────────────────────────────
    def _resolve_targets(
        self,
        parsed: ParsedRawOutput,
        warnings: list[str],
    ) -> list[str]:
        """target_chapters_or_concepts 정규화 (옵션 B)."""
        raw = parsed.target_chapters_or_concepts
        if not raw:
            warnings.append(f"target_chapters_or_concepts: defaults {len(TARGET_CHAPTERS)}개 모듈 사용")
            return list(TARGET_CHAPTERS)

        normalized, rejected = normalize_target_list(raw)
        if rejected:
            warnings.append(
                f"target_chapters_or_concepts: 정규화 실패한 항목 {rejected} 제외"
            )
        if not normalized:
            warnings.append("target_chapters_or_concepts: 유효한 항목 없음 → defaults 사용")
            return list(TARGET_CHAPTERS)
        return normalized

    def _resolve_positive_int(
        self,
        value: Optional[int],
        default: int,
        field_name: str,
        warnings: list[str],
    ) -> int:
        if value is None:
            warnings.append(f"{field_name}: defaults {default} 사용")
            return default
        if not isinstance(value, int) or value <= 0:
            warnings.append(
                f"{field_name}: 잘못된 값 '{value}' → defaults {default} 사용"
            )
            return default
        return value

    def _resolve_evaluation_focus(
        self,
        parsed: ParsedRawOutput,
        warnings: list[str],
    ) -> EvaluationFocus:
        raw = parsed.evaluation_focus
        if raw is None:
            return EvaluationFocus(EXAM_BASICS["evaluation_focus"])
        try:
            return EvaluationFocus(raw)
        except ValueError:
            warnings.append(
                f"evaluation_focus: 알 수 없는 값 '{raw}' → defaults 'balanced' 사용"
            )
            return EvaluationFocus(EXAM_BASICS["evaluation_focus"])

    def _resolve_difficulty_group(
        self,
        parsed: ParsedRawOutput,
        warnings: list[str],
    ) -> DifficultyDistributionGroup:
        raw = parsed.difficulty_distribution_group
        if raw is None:
            warnings.append(
                f"difficulty_distribution_group: defaults {DIFFICULTY_DISTRIBUTION_GROUP} 사용"
            )
            return DifficultyDistributionGroup(**DIFFICULTY_DISTRIBUTION_GROUP)
        try:
            return DifficultyDistributionGroup.model_validate(raw)
        except ValidationError as exc:
            warnings.append(
                f"difficulty_distribution_group: 검증 실패 ({exc.error_count()}건) → defaults 사용"
            )
            return DifficultyDistributionGroup(**DIFFICULTY_DISTRIBUTION_GROUP)

    def _resolve_type_distribution(
        self,
        parsed: ParsedRawOutput,
        warnings: list[str],
    ) -> QuestionTypeDistribution:
        raw = parsed.question_type_distribution
        if raw is None:
            warnings.append("question_type_distribution: defaults 사용")
            return QuestionTypeDistribution(**_default_question_type_distribution())
        try:
            return QuestionTypeDistribution.model_validate(raw)
        except ValidationError as exc:
            warnings.append(
                f"question_type_distribution: 검증 실패 ({exc.error_count()}건) → defaults 사용"
            )
            return QuestionTypeDistribution(**_default_question_type_distribution())

    def _check_distribution_consistency(
        self,
        diff_group: DifficultyDistributionGroup,
        type_dist: QuestionTypeDistribution,
        warnings: list[str],
    ) -> None:
        diff_sum = diff_group.easy + diff_group.medium + diff_group.hard
        type_sum = (
            type_dist.MCQ_single
            + type_dist.short_answer
            + type_dist.long_answer
            + type_dist.case_analysis
        )
        if diff_sum != type_sum:
            warnings.append(
                f"분포 합계 불일치: difficulty {diff_sum} vs type {type_sum} "
                f"— ExamPlanner가 조정 시도"
            )


# ────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import doctest
    failed, total = doctest.testmod(verbose=False)
    print(f"✓ doctests: {total - failed}/{total} passed")

    # 정규화 추가 케이스
    cases = [
        ("M1_1", "M1_1"),
        ("m1.1", "M1_1"),
        ("M1-1", "M1_1"),
        ("M2_1_2_KJ_Method", "M2_1_2_kj_method"),
        ("manufacturing_overview", "manufacturing_overview"),
        ("MANUFACTURING_OVERVIEW", "manufacturing_overview"),
    ]
    for raw, expected in cases:
        actual = normalize_target(raw)
        mark = "✓" if actual == expected else "✗"
        print(f"  {mark} {raw!r} → {actual!r} (expected {expected!r})")
