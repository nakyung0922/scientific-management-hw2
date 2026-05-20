"""
페이지 묶음 → 의미 단위 청크.

슬라이드 1장의 텍스트가 너무 짧아서 그것만으로는 '개념' 을 형성하기 어려운
경우가 많음. 인접 N 페이지를 묶어 하나의 청크 후보로 만들고, 너무 짧은
청크는 다음 청크에 머지하는 단순 규칙을 적용.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agents.material_collector.pdf_reader import PageRecord


@dataclass
class Chunk:
    """LLM 요약에 들어갈 한 단위.

    page_range: (start_page, end_page) — inclusive
    """
    text: str
    page_range: tuple[int, int]
    title: Optional[str]
    methods: list[str]   # 청크에 포함된 페이지들의 추출 방법 (text/ocr)

    @property
    def has_ocr(self) -> bool:
        return "ocr" in self.methods

    @property
    def page_labels(self) -> list[str]:
        """ConceptNode.source_pages 에 들어갈 라벨 (p.숫자 형식).

        실제 module_id 접두는 agent.py 에서 결합.
        """
        start, end = self.page_range
        if start == end:
            return [f"p.{start}"]
        return [f"p.{p}" for p in range(start, end + 1)]


def make_chunks(
    pages: list[PageRecord],
    *,
    pages_per_chunk: int = 2,
    min_chunk_chars: int = 80,
) -> list[Chunk]:
    """페이지 리스트를 청크로 묶는다.

    Args:
        pages: PDF 한 파일분 페이지 (extract_pages 결과)
        pages_per_chunk: 한 청크에 묶을 페이지 수 (기본 2)
        min_chunk_chars: 이 글자 수 미만인 청크는 다음 청크와 머지 (마지막 청크면 이전과 머지)

    Returns:
        list[Chunk]: 의미 단위 청크 리스트
    """
    if not pages:
        return []

    # 1) 단순 윈도우링
    raw_chunks: list[Chunk] = []
    i = 0
    while i < len(pages):
        window = pages[i : i + pages_per_chunk]
        if not window:
            break
        text = "\n\n".join(p.text for p in window)
        title = next((p.title for p in window if p.title), None)
        raw_chunks.append(
            Chunk(
                text=text,
                page_range=(window[0].page_no, window[-1].page_no),
                title=title,
                methods=[p.method for p in window],
            )
        )
        i += pages_per_chunk

    # 2) 짧은 청크 머지 (앞 청크에 흡수)
    merged: list[Chunk] = []
    for ch in raw_chunks:
        if merged and len(ch.text) < min_chunk_chars:
            prev = merged[-1]
            merged[-1] = Chunk(
                text=prev.text + "\n\n" + ch.text,
                page_range=(prev.page_range[0], ch.page_range[1]),
                title=prev.title or ch.title,
                methods=prev.methods + ch.methods,
            )
        else:
            merged.append(ch)

    # 3) 첫 청크가 너무 짧으면 (드물지만) 두 번째와 머지
    if len(merged) >= 2 and len(merged[0].text) < min_chunk_chars:
        first, second = merged[0], merged[1]
        merged[0] = Chunk(
            text=first.text + "\n\n" + second.text,
            page_range=(first.page_range[0], second.page_range[1]),
            title=first.title or second.title,
            methods=first.methods + second.methods,
        )
        merged.pop(1)

    return merged
