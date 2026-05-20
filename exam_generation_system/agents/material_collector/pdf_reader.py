"""
PDF 텍스트 추출 (PyMuPDF + Tesseract OCR fallback).

설계 결정 (팀 회의):
    - 1차: PyMuPDF (`fitz`) 로 직접 텍스트 추출
    - 2차: 페이지가 이미지 기반 PDF 이면 Tesseract 로 OCR (lang=kor+eng)
    - 강의자료가 PPT export PDF 인 경우 보통 100% OCR 경로로 감.

함수는 부수효과 없이 페이지 단위 dict 리스트만 반환.
청킹·요약은 다음 모듈 (chunker.py, summarizer.py) 책임.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# OCR 은 선택적 의존성. 없으면 OCR fallback 만 비활성화.
try:
    import io
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


# ────────────────────────────────────────────────────────────
# 노이즈 정리
# ────────────────────────────────────────────────────────────
_FOOTER_PATTERNS = [
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),       # "10 / 22"
    re.compile(r"^\s*\d+\s*$"),                  # "10"
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
]


def _clean_line(line: str) -> str:
    return line.replace("\u200b", "").strip()


def _is_noise(line: str) -> bool:
    if not line:
        return True
    return any(p.match(line) for p in _FOOTER_PATTERNS)


def _clean_page(raw: str) -> str:
    """페이지 텍스트의 페이지번호/빈줄 노이즈 제거."""
    lines = [_clean_line(l) for l in raw.splitlines()]
    lines = [l for l in lines if not _is_noise(l)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_title(text: str) -> Optional[str]:
    """페이지의 첫 번째 짧은 라인을 슬라이드 제목으로 추정."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if 1 < len(line) <= 80:
            return line
        return None
    return None


# ────────────────────────────────────────────────────────────
# OCR
# ────────────────────────────────────────────────────────────
def _ocr_page(page: "fitz.Page", lang: str, dpi: int) -> str:
    """페이지를 이미지로 렌더링한 뒤 Tesseract OCR."""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=lang)


# ────────────────────────────────────────────────────────────
# 메인 진입점
# ────────────────────────────────────────────────────────────
class PageRecord:
    """페이지 한 장의 추출 결과."""
    __slots__ = ("page_no", "title", "text", "method")

    def __init__(self, page_no: int, title: Optional[str], text: str, method: str):
        self.page_no = page_no
        self.title = title
        self.text = text
        self.method = method  # 'text' | 'ocr'

    def __repr__(self) -> str:
        return f"<PageRecord p.{self.page_no} method={self.method} len={len(self.text)}>"


def extract_pages(
    pdf_path: str | Path,
    *,
    enable_ocr: bool = True,
    ocr_lang: str = "kor+eng",
    ocr_dpi: int = 200,
    ocr_threshold_chars: int = 20,
) -> list[PageRecord]:
    """PDF 한 파일에서 페이지 단위로 텍스트 추출.

    Args:
        pdf_path: PDF 파일 경로
        enable_ocr: OCR fallback 사용 여부 (기본 True)
        ocr_lang: Tesseract 언어 코드 (예: 'kor+eng')
        ocr_dpi: OCR 렌더링 DPI (200 이 균형점)
        ocr_threshold_chars: 텍스트가 이 글자 수 미만이면 OCR 시도

    Returns:
        list[PageRecord]: 비어있지 않은 페이지만 (text 가 있어야 함)
    """
    pdf_path = str(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    use_ocr = enable_ocr and _OCR_AVAILABLE
    if enable_ocr and not _OCR_AVAILABLE:
        logger.warning(
            "OCR 요청됐지만 pytesseract/Pillow 가 없어 비활성화됨. "
            "이미지 기반 PDF 는 빈 결과가 나올 수 있음."
        )

    records: list[PageRecord] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            cleaned = _clean_page(raw)
            method = "text"

            if len(cleaned) < ocr_threshold_chars and use_ocr:
                try:
                    ocr_raw = _ocr_page(page, lang=ocr_lang, dpi=ocr_dpi)
                    cleaned = _clean_page(ocr_raw)
                    method = "ocr"
                except Exception as e:
                    logger.warning(
                        "OCR 실패 page=%d file=%s err=%s", idx, pdf_path, e
                    )

            if not cleaned:
                continue
            records.append(
                PageRecord(
                    page_no=idx,
                    title=_guess_title(cleaned),
                    text=cleaned,
                    method=method,
                )
            )

    return records


def ocr_available() -> bool:
    """OCR 의존성이 갖춰져있는지."""
    return _OCR_AVAILABLE
