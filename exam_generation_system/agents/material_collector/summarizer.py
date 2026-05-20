"""
청크 → 개념 요약 (Gemini via PG2 GeminiClient).

PG2 의 GeminiClient 를 그대로 사용하므로:
    - response_mime_type=application/json 강제됨
    - 재시도 + 사용량 트래커 자동
    - mock 모드로 단위 테스트 가능

이 모듈은 LLM 호출의 래퍼만 담당. 청크 만들기 / concept_id 결정은
agent.py 가 조율한다.
"""
from __future__ import annotations

import logging
from typing import Any

from common.gemini_client import GeminiClient

from agents.material_collector.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "Material_Collector"


class ConceptDraft:
    """LLM 이 청크 하나에서 뽑아낸 raw concept (concept_id 미부여)."""
    __slots__ = ("concept_name", "summary", "keywords", "is_empty", "from_fallback")

    def __init__(
        self,
        concept_name: str,
        summary: str,
        keywords: list[str],
        is_empty: bool,
        from_fallback: bool = False,
    ):
        self.concept_name = concept_name
        self.summary = summary
        self.keywords = keywords
        self.is_empty = is_empty
        self.from_fallback = from_fallback


def summarize_chunk(
    client: GeminiClient,
    *,
    course_name: str,
    source_file: str,
    page_range: tuple[int, int],
    section_title: str | None,
    passage: str,
) -> ConceptDraft:
    """청크 하나에 대해 LLM 호출.

    실패 (LLM error, JSON 파싱 실패) 시 빈 draft 반환 (서비스 중단 방지).
    """
    system_prompt = SYSTEM_PROMPT.format(course_name=course_name)
    user_prompt = build_user_prompt(
        source_file=source_file,
        page_range=page_range,
        section_title=section_title,
        passage=passage,
    )

    try:
        result: dict[str, Any] = client.generate_json(
            agent_name=AGENT_NAME,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,  # 결정적 추출
        )
    except Exception as e:
        logger.warning(
            "LLM 호출 실패 (%s p.%s): %s — 빈 draft 반환",
            source_file, page_range, e,
        )
        return ConceptDraft("", "", [], is_empty=True, from_fallback=True)

    # 응답 검증 + 기본값
    concept_name = (result.get("concept_name") or "").strip()
    summary = (result.get("summary") or "").strip()
    keywords_raw = result.get("keywords") or []
    if not isinstance(keywords_raw, list):
        keywords_raw = []
    keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
    is_empty = bool(result.get("is_empty", False))

    # LLM 이 is_empty=False 라고 답했지만 실제로는 비어있을 수도 있음
    if not concept_name and not summary:
        is_empty = True

    return ConceptDraft(
        concept_name=concept_name,
        summary=summary,
        keywords=keywords,
        is_empty=is_empty,
        from_fallback=False,
    )
