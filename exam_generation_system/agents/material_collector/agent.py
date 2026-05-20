"""
Material Collector agent — 메인 클래스.

흐름:
    PDFs
      ↓ pdf_reader.extract_pages()   (PyMuPDF + OCR fallback)
    PageRecord[]
      ↓ chunker.make_chunks()        (페이지 묶음 → 의미 청크)
    Chunk[]
      ↓ summarizer.summarize_chunk() (Gemini → ConceptDraft)
    ConceptDraft[]
      ↓ concept_id.make_concept_id() (v0.5 네이밍 + 충돌 회피)
    ConceptUnit[]
      ↓ MaterialCollectorOutput      (Pydantic, StrictBase)
      ↓ envelope.wrap_payload()      (MessageEnvelope)
    → Topic_Analyzer 가 수신

v0.5 명세 (agentic_system_schema_v0.5.json) 의:
    - global_message_schema      → MessageEnvelope (PG2 common.schemas)
    - concept_id_naming_rule     → concept_id.make_concept_id
    - LLM_CONFIG.provider        → vertex_ai (PG2 GeminiClient)
    - target_chapters            → 시험 범위 외 자료 warning 표시
    - routing_status='flow'      → 정상 흐름
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from common.envelope import wrap_payload
from common.enums import RoutingStatus
from common.gemini_client import GeminiClient
from common.schemas import MessageEnvelope
from config.defaults import EXAM_HEADER_TEMPLATE

from agents.material_collector.chunker import Chunk, make_chunks
from agents.material_collector.concept_id import (
    label_module_from_filename,
    make_concept_id,
    slugify,
)
from agents.material_collector.pdf_reader import (
    PageRecord,
    extract_pages,
    ocr_available,
)
from agents.material_collector.schemas import (
    ConceptUnit,
    MaterialCollectorOutput,
)
from agents.material_collector.summarizer import ConceptDraft, summarize_chunk

logger = logging.getLogger(__name__)

AGENT_NAME = "Material_Collector"
NEXT_AGENT = "Topic_Analyzer"


# ────────────────────────────────────────────────────────────
# Agent 설정 dataclass
# ────────────────────────────────────────────────────────────
@dataclass
class MaterialCollectorConfig:
    """Agent 실행 설정. 기본값은 모두 안전한 값."""

    course_name: str = field(
        default_factory=lambda: EXAM_HEADER_TEMPLATE["course_name"]
    )
    pages_per_chunk: int = 2
    min_chunk_chars: int = 80

    # OCR
    enable_ocr: bool = True
    ocr_lang: str = "kor+eng"
    ocr_dpi: int = 200

    # LLM
    use_llm: bool = True
    """False 면 LLM 호출을 아예 안 하고, concept_name 을 휴리스틱(첫 줄)으로
    채워서 빈 출력으로 가지 않게 함. 단위 테스트/오프라인 디버깅용."""

    # concept_id 생성
    max_slug_words: int = 4


# ────────────────────────────────────────────────────────────
# Agent
# ────────────────────────────────────────────────────────────
class MaterialCollector:
    """Material Collector agent.

    사용 예:
        client = GeminiClient(mock=False)  # 또는 mock=True
        agent = MaterialCollector(client=client)
        env = agent.run(
            pdf_paths=["M1_4_taylorism.pdf"],
            session_id="session_001",
        )
        # env 를 Topic_Analyzer 로 전달
    """

    def __init__(
        self,
        client: GeminiClient,
        config: Optional[MaterialCollectorConfig] = None,
    ):
        self.client = client
        self.config = config or MaterialCollectorConfig()

    # ------------------------------------------------------------------
    def run(
        self,
        pdf_paths: list[str | Path],
        session_id: str,
    ) -> MessageEnvelope:
        """파이프라인 실행 → MessageEnvelope.

        실패한 파일은 warnings 에 기록하고 건너뜀 (전체 중단 안 함).
        """
        all_concepts: list[ConceptUnit] = []
        used_ids: set[str] = set()
        used_module_ids: set[str] = set()
        file_to_module: dict[str, str] = {}  # 사용자에게 알려줄 매핑
        warnings: list[str] = []
        method_counts: dict[str, int] = {"text": 0, "ocr": 0}
        processed_files: list[str] = []
        total_pages = 0

        for pdf_path in pdf_paths:
            pdf_path = str(pdf_path)
            basename = os.path.basename(pdf_path)

            # 1) 페이지 추출 ----------------------------------------
            try:
                pages = extract_pages(
                    pdf_path,
                    enable_ocr=self.config.enable_ocr,
                    ocr_lang=self.config.ocr_lang,
                    ocr_dpi=self.config.ocr_dpi,
                )
            except FileNotFoundError:
                warnings.append(f"file_not_found: {basename}")
                continue
            except Exception as e:
                warnings.append(f"extract_failed: {basename} ({e})")
                continue

            if not pages:
                warnings.append(f"no_text_extracted: {basename}")
                continue

            for p in pages:
                method_counts[p.method] = method_counts.get(p.method, 0) + 1
            total_pages += len(pages)
            processed_files.append(basename)

            # 2) module_id 라벨링 (파일별 충돌 회피) -----------------
            # Material Collector 는 시험 범위에 대해 가정하지 않는다.
            # 어떤 파일이 들어와도 안정적인 라벨을 부여한다.
            module_id = label_module_from_filename(
                basename,
                used_labels=used_module_ids,
            )
            used_module_ids.add(module_id)
            file_to_module[basename] = module_id

            # ASCII 가 없는 파일명은 'material_<hash>' 형태로 fallback 되므로
            # 호출자가 알 수 있도록 info 성 warning 을 남긴다 (에러는 아님)
            if module_id.startswith("material_"):
                warnings.append(
                    f"auto_labeled: {basename} → {module_id} "
                    "(파일명에 ASCII 영문이 없어 hash 기반 라벨 부여)"
                )

            # 3) 청킹 -----------------------------------------------
            chunks = make_chunks(
                pages,
                pages_per_chunk=self.config.pages_per_chunk,
                min_chunk_chars=self.config.min_chunk_chars,
            )

            # 4) 청크별 LLM 요약 → ConceptUnit ---------------------
            file_concepts = self._process_chunks(
                chunks=chunks,
                module_id=module_id,
                source_file=basename,
                used_ids=used_ids,
                warnings=warnings,
            )
            all_concepts.extend(file_concepts)

        # 5) Payload 구성 ------------------------------------------
        # OCR 사용했는데 ocr_languages 비어있으면 채워주기
        ocr_langs: list[str] = []
        if method_counts.get("ocr", 0) > 0:
            ocr_langs = [s.strip() for s in self.config.ocr_lang.split("+")]

        payload = MaterialCollectorOutput(
            session_id=session_id,
            course_name=self.config.course_name,
            source_files=processed_files,
            total_pages=total_pages,
            total_concepts=len(all_concepts),
            concepts=all_concepts,
            file_to_module_id=file_to_module,
            extraction_methods={k: v for k, v in method_counts.items() if v > 0},
            ocr_languages=ocr_langs,
            warnings=warnings,
        )

        # 6) Envelope 으로 감싸서 반환 ------------------------------
        return wrap_payload(
            session_id=session_id,
            source_agent=AGENT_NAME,
            target_agent=NEXT_AGENT,
            payload=payload,
            routing_status=RoutingStatus.FLOW,
        )

    # ------------------------------------------------------------------
    def _process_chunks(
        self,
        *,
        chunks: list[Chunk],
        module_id: str,
        source_file: str,
        used_ids: set[str],
        warnings: list[str],
    ) -> list[ConceptUnit]:
        """청크 리스트를 ConceptUnit 리스트로."""
        out: list[ConceptUnit] = []

        for chunk in chunks:
            if self.config.use_llm:
                draft = summarize_chunk(
                    self.client,
                    course_name=self.config.course_name,
                    source_file=source_file,
                    page_range=chunk.page_range,
                    section_title=chunk.title,
                    passage=chunk.text,
                )
            else:
                # LLM 우회 휴리스틱 (offline test 용)
                draft = self._heuristic_draft(chunk)

            if draft.is_empty or not draft.concept_name:
                # 표지/목차 등 — 청크 스킵 (warning 도 남기지 않음, 정상 케이스)
                continue

            # v0.5: concept_id 는 영문/숫자/_ 만 허용 → concept_name 에 ASCII 영어가
            # 없으면 slug 가 빈 문자열이 되므로 미리 거르고 warning.
            if not slugify(draft.concept_name, max_words=self.config.max_slug_words):
                warnings.append(
                    f"concept_name_not_ascii: {source_file} p.{chunk.page_range} "
                    f"name={draft.concept_name!r}"
                )
                continue

            # concept_id 생성 (충돌 회피)
            cid = make_concept_id(
                module_id=module_id,
                concept_name=draft.concept_name,
                used_ids=used_ids,
                max_slug_words=self.config.max_slug_words,
            )
            used_ids.add(cid)

            # source_pages: ConceptNode.source_pages 형식 (e.g. "M1_4 p.10")
            source_pages = [f"{module_id} {label}" for label in chunk.page_labels]

            # confidence: OCR 페이지가 섞이면 낮춤, LLM fallback 이면 더 낮춤
            confidence = 1.0
            if chunk.has_ocr:
                confidence *= 0.7
            if draft.from_fallback:
                confidence *= 0.5

            try:
                unit = ConceptUnit(
                    concept_id=cid,
                    concept_name=draft.concept_name,
                    summary=draft.summary,
                    keywords=draft.keywords,
                    source_file=source_file,
                    source_pages=source_pages,
                    raw_text=chunk.text,
                    section_title=chunk.title,
                    extraction_confidence=round(confidence, 3),
                )
            except Exception as e:
                # Pydantic validation 실패 (concept_id 형식 등)
                warnings.append(
                    f"concept_unit_invalid: {source_file} p.{chunk.page_range} ({e})"
                )
                continue

            out.append(unit)

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_draft(chunk: Chunk) -> ConceptDraft:
        """LLM 없이 청크에서 간단한 draft 생성 (offline 테스트용).

        - concept_name: 첫 짧은 줄
        - summary: 처음 2문장
        - keywords: 4자 이상 빈도 상위 토큰
        """
        import re

        lines = [l.strip() for l in chunk.text.splitlines() if l.strip()]
        concept_name = ""
        for l in lines:
            if 2 < len(l) <= 60:
                concept_name = l
                break

        sentences = re.split(r"(?<=[\.!\?。])\s+", chunk.text)
        summary = " ".join(s.strip() for s in sentences[:2] if s.strip())[:400]

        tokens = re.findall(r"[A-Za-z가-힣]{4,}", chunk.text)
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        keywords = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:5]]

        return ConceptDraft(
            concept_name=concept_name,
            summary=summary,
            keywords=keywords,
            is_empty=not concept_name,
            from_fallback=True,
        )
