"""
Material Collector — Agent 1-1 (PG1, R3).

입력: PDF 파일 경로(들)
출력: list[ConceptSummary] (Topic_Analyzer 입력)

설계 원칙:
- Pass 1 (Python, pypdf): 페이지별 raw text 추출 → RawDocument
- Pass 2 (LLM light): 모듈 단위 요약 + 핵심 phrase → ConceptSummary
- module_id는 파일명 휴리스틱으로 추정 (예: "M1_1_What_is_Work" → "M1_1")
- LLM 호출은 모듈당 1회 (light 모델, temperature=0)

LOA = 2 (자동 처리, 사람 개입 거의 없음).
"""
from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import Field, ValidationError

from common.gemini_client import GeminiClient
from common.schemas import (
    ConceptSummary,
    PageText,
    RawDocument,
    StrictBase,
)
from config.defaults import PROJECT_ROOT

logger = logging.getLogger(__name__)

AGENT_NAME = "Material_Collector"
PROMPT_REL_PATH = "config/prompts/material_collector.md"


# ────────────────────────────────────────────────────────────
# LLM raw output 검증 모델
# ────────────────────────────────────────────────────────────
class SummaryRawOutput(StrictBase):
    """LLM이 반환하는 raw JSON 구조."""
    title: str
    summary_text: str
    key_phrases: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────
# 프롬프트 캐시
# ────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _load_prompt_file(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


# ────────────────────────────────────────────────────────────
# 파일명 → module_id 추정 휴리스틱
# ────────────────────────────────────────────────────────────
_MODULE_ID_PATTERNS = [
    # "M1_2_Why_Work_Matters_260312_8092103.pdf" → "M1_2"
    # "M2_1_5_kj_method.pdf" → "M2_1_5"
    re.compile(r"^(M\d+(?:_\d+){1,3})", re.IGNORECASE),
    # "manufacturing_overview.pdf" → 그대로 stem 사용
]


def infer_module_id(filename: str) -> str:
    """파일명에서 v0.5 네이밍 규칙에 맞는 module_id 추정.

    >>> infer_module_id("M1_2_Why_Work_Matters_260312.pdf")
    'M1_2'
    >>> infer_module_id("M2_1_2_Understanding_KJ_Method.pdf")
    'M2_1_2'
    >>> infer_module_id("manufacturing_overview.pdf")
    'manufacturing_overview'
    """
    stem = Path(filename).stem

    for pattern in _MODULE_ID_PATTERNS:
        m = pattern.match(stem)
        if m:
            # M prefix만 대문자 허용, 나머지 소문자화 (v0.5 네이밍 규칙)
            mid = m.group(1)
            return mid[0].upper() + mid[1:].lower() if mid[0].lower() == "m" else mid.lower()

    # 패턴 매치 실패 → 파일명을 소문자화하고 공백·점은 _로
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", stem.lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown_module"


# ────────────────────────────────────────────────────────────
# 페이지 텍스트 정제
# ────────────────────────────────────────────────────────────
def clean_page_text(raw: str) -> str:
    """페이지 추출 텍스트의 명백한 잡음 제거.

    - 줄 끝 공백, 다중 빈 줄 정리
    - 페이지 번호 단독 라인 제거
    - 매우 짧은(2자 이하) 라인 제거

    원문 의미는 건드리지 않음. LLM이 보기 좋게만 정리.
    """
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # 페이지 번호 단독 라인 (숫자만)
        if stripped.isdigit() and len(stripped) <= 3:
            continue
        # 너무 짧은 라인 (의미 없는 fragment)
        if len(stripped) <= 2:
            continue
        lines.append(stripped)
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# Pass 1: pypdf로 RawDocument 만들기
# ────────────────────────────────────────────────────────────
def parse_pdf_to_raw_document(pdf_path: Path) -> RawDocument:
    """단일 PDF → RawDocument.

    pypdf로 페이지별 텍스트 추출 후 clean_page_text로 잡음 제거.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pypdf가 설치되지 않았습니다. pip install pypdf 를 실행하세요."
        ) from e

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일 없음: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "[Material_Collector] %s page %d extract failed: %s",
                pdf_path.name, i, exc,
            )
            raw = ""
        cleaned = clean_page_text(raw)
        if cleaned:  # 빈 페이지는 스킵
            pages.append(PageText(page_no=i, text=cleaned))

    if not pages:
        # 모든 페이지가 비어있어도 최소 placeholder 1개는 둠
        pages = [PageText(page_no=1, text="(no extractable text)")]

    return RawDocument(
        module_id=infer_module_id(pdf_path.name),
        source_filename=pdf_path.name,
        pages=pages,
    )


# ────────────────────────────────────────────────────────────
# 메인 Collector 클래스
# ────────────────────────────────────────────────────────────
class MaterialCollector:
    """Material Collector Agent.

    Usage:
        collector = MaterialCollector(gemini_client=GeminiClient(mock=True))
        summaries = collector.collect([Path("M1_1.pdf"), Path("M1_2.pdf")])
    """

    MIN_KEY_PHRASES = 5
    MAX_KEY_PHRASES = 15

    def __init__(self, client: GeminiClient) -> None:
        self.client = client
        self.system_prompt = _load_prompt_file(str(PROJECT_ROOT / PROMPT_REL_PATH))

    # ── 공개 인터페이스 ──────────────────────────────────────
    def collect(
        self,
        pdf_paths: list[Path],
    ) -> list[ConceptSummary]:
        """여러 PDF를 받아 ConceptSummary 리스트 반환.

        각 PDF는 1개의 ConceptSummary를 생성. 1 LLM call per PDF.
        실패한 PDF는 빈 summary + warnings로 fallback (전체 흐름 멈추지 않음).
        """
        summaries: list[ConceptSummary] = []
        for path in pdf_paths:
            try:
                raw_doc = parse_pdf_to_raw_document(path)
                summary = self._summarize_one(raw_doc)
                summaries.append(summary)
            except Exception as exc:
                logger.warning(
                    "[Material_Collector] failed on %s: %s", path, exc,
                )
                summaries.append(self._fallback_summary(path, str(exc)))
        return summaries

    # ── 단일 PDF 요약 ───────────────────────────────────────
    def _summarize_one(self, raw_doc: RawDocument) -> ConceptSummary:
        warnings: list[str] = []
        user_prompt = self._build_user_prompt(raw_doc)

        try:
            raw_json = self.client.generate_json(
                agent_name=AGENT_NAME,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
            )
            validated = SummaryRawOutput.model_validate(raw_json)
        except (ValidationError, RuntimeError) as exc:
            logger.warning(
                "[Material_Collector] %s LLM failed: %s. Using fallback.",
                raw_doc.source_filename, exc,
            )
            return self._fallback_summary_from_doc(raw_doc, str(exc))

        # key_phrases 개수 검증 → warning만, 흐름 중단 안 함
        n_phrases = len(validated.key_phrases)
        if n_phrases < self.MIN_KEY_PHRASES:
            warnings.append(
                f"key_phrases가 부족함: {n_phrases}개 (권장 {self.MIN_KEY_PHRASES}개 이상)"
            )
        elif n_phrases > self.MAX_KEY_PHRASES:
            warnings.append(
                f"key_phrases가 너무 많음: {n_phrases}개. 상위 {self.MAX_KEY_PHRASES}개로 자름."
            )
            validated.key_phrases = validated.key_phrases[: self.MAX_KEY_PHRASES]

        return ConceptSummary(
            module_id=raw_doc.module_id,
            source_filename=raw_doc.source_filename,
            title=validated.title,
            summary_text=validated.summary_text,
            key_phrases=validated.key_phrases,
            page_count=len(raw_doc.pages),
            warnings=warnings,
        )

    def _build_user_prompt(self, raw_doc: RawDocument) -> str:
        """LLM에 넘길 user prompt 구성."""
        payload = {
            "module_id": raw_doc.module_id,
            "source_filename": raw_doc.source_filename,
            "pages": [
                {"page_no": p.page_no, "text": p.text}
                for p in raw_doc.pages
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    # ── Fallback (LLM·파싱 실패 시) ──────────────────────────
    def _fallback_summary(self, path: Path, error_msg: str) -> ConceptSummary:
        """PDF 파싱 자체가 실패했을 때 (FileNotFoundError 등)."""
        return ConceptSummary(
            module_id=infer_module_id(path.name),
            source_filename=path.name,
            title=path.stem,
            summary_text="(PDF parsing failed; no content available)",
            key_phrases=[],
            page_count=1,
            warnings=[f"parse_failure: {error_msg}"],
        )

    def _fallback_summary_from_doc(
        self,
        raw_doc: RawDocument,
        error_msg: str,
    ) -> ConceptSummary:
        """파싱은 성공했지만 LLM 호출이 실패했을 때."""
        # 최소한의 정보로 fallback: 첫 페이지 텍스트 일부를 summary로
        first_text = raw_doc.pages[0].text if raw_doc.pages else ""
        truncated = first_text[:300] if first_text else "(no text)"
        return ConceptSummary(
            module_id=raw_doc.module_id,
            source_filename=raw_doc.source_filename,
            title=raw_doc.module_id,
            summary_text=truncated,
            key_phrases=[],
            page_count=len(raw_doc.pages),
            warnings=[f"llm_failure: {error_msg}", "fallback_used"],
        )


# ────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import doctest
    failed, total = doctest.testmod(verbose=False)
    print(f"✓ doctests: {total - failed}/{total} passed")

    # module_id 추정 추가 케이스
    cases = [
        ("M1_1_What_is_Work_Revised_260312.pdf", "M1_1"),
        ("M2_1_2_Understanding_and_Structuring_Problems_1_KJ_Method.pdf", "M2_1_2"),
        ("M3_1_1_motion.pdf", "M3_1_1"),
        ("manufacturing_overview.pdf", "manufacturing_overview"),
        ("random_file.pdf", "random_file"),
    ]
    for filename, expected in cases:
        actual = infer_module_id(filename)
        mark = "✓" if actual == expected else "✗"
        print(f"  {mark} {filename!r} → {actual!r} (expected {expected!r})")
