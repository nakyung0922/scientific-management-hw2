"""
concept_id 생성 유틸 (v0.5 네이밍 규칙).

규칙 (`agentic_system_schema_v0.5.json` 의 concept_id_naming_rule):
    format   : {module_id}_{snake_case_slug}
    examples : M2_1_5_kj_method, M3_1_1_therbligs, manufacturing_overview
    rules    :
      - 점(.) 은 언더스코어(_) 로 변환
      - 영문 소문자 + 숫자 + _ 만 허용 (module prefix 의 M 은 대문자 허용)
      - slug 최대 4 단어
      - 충돌 시 _v2 추가

이 모듈은:
    1) 파일명에서 module_id 추론   (e.g. "M1.4_Taylorism.pdf" → "M1_4")
    2) 자유 문자열에서 snake_case_slug 생성 (e.g. "Taylor's 4 Principles" → "taylor_4_principles")
    3) 충돌 회피 (집합 내에서 _v2, _v3 부여)
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

from config.defaults import TARGET_CHAPTERS


# ────────────────────────────────────────────────────────────
# module_id 추론
# ────────────────────────────────────────────────────────────

# "M5_1", "M5.1", "M5-1", "M2_1_5" 같은 패턴을 인식
_MODULE_PATTERN = re.compile(r"M\d+(?:[._-]\d+)*", re.IGNORECASE)


def infer_module_id_from_filename(filename: str) -> str:
    """파일명에서 module_id 추출.

    >>> infer_module_id_from_filename("M5_1_Automation.pdf")
    'M5_1'
    >>> infer_module_id_from_filename("M2.1.5_KJ_method.pdf")
    'M2_1_5'
    >>> infer_module_id_from_filename("manufacturing_overview.pdf")
    'manufacturing_overview'

    파일명에서 M{숫자}.{숫자}... 패턴을 찾지 못하면 stem 전체를 slug 화하여 반환.
    이는 'manufacturing_overview' 같이 TARGET_CHAPTERS 에 들어있는 이름이
    그대로 module_id 로 쓰일 수 있게 함.
    """
    stem = Path(filename).stem
    match = _MODULE_PATTERN.search(stem)
    if match:
        # 점/하이픈을 언더스코어로 통일하고 M 은 대문자 유지
        token = match.group(0)
        normalized = re.sub(r"[.\-]", "_", token)
        # M 은 대문자 유지, 나머지는 그대로
        return "M" + normalized[1:]
    # fallback: TARGET_CHAPTERS 에 있는 이름이면 그대로 사용
    slug = slugify(stem)
    if slug in TARGET_CHAPTERS:
        return slug
    return slug or "unknown_module"


def module_id_in_target_chapters(module_id: str) -> bool:
    """module_id 가 시험 범위 안에 있는지.

    Material Collector 가 입력 PDF 중 시험 범위 밖 자료를 처리할 때
    warning 을 남기기 위한 헬퍼.
    """
    # 정확 일치 또는 prefix 일치 (M2_1 은 M2_1_1 등을 포함)
    if module_id in TARGET_CHAPTERS:
        return True
    for ch in TARGET_CHAPTERS:
        if ch.startswith(module_id + "_") or module_id.startswith(ch + "_"):
            return True
    return False


# ────────────────────────────────────────────────────────────
# slug 화
# ────────────────────────────────────────────────────────────

# 의미 없는 토큰 (영어 stop words + 한국어 조사 일부)
_STOP_WORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for",
    "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "as", "this", "that", "these", "those", "it", "its",
    "에", "에서", "은", "는", "이", "가", "을", "를", "의", "와", "과",
}


def slugify(text: str, max_words: int = 4) -> str:
    """자유 문자열을 v0.5 규칙의 snake_case slug 로.

    v0.5 명세: 영문 소문자 + 숫자 + _ 만 허용.
    한글/한자/일본어 등 비 ASCII 문자는 transliterate 가 비결정적이라
    여기서는 그냥 버린다. 영어로 추출하는 책임은 LLM 프롬프트가 진다
    (`concept_name MUST be ENGLISH` 강제).

    >>> slugify("Taylor's 4 Principles")
    'taylors_4_principles'
    >>> slugify("KJ Method")
    'kj_method'
    >>> slugify("자동화의 정의")
    ''
    >>> slugify("KJ Method (브레인스토밍 정리)")
    'kj_method'
    >>> slugify("a very long title with too many tokens here please", max_words=3)
    'very_long_title'
    """
    if not text:
        return ""

    # 유니코드 정규화 (NFC)
    text = unicodedata.normalize("NFC", text)

    # 어퍼스트로피 류 제거
    text = text.replace("'", "").replace("'", "").replace("'", "")

    # ASCII 영문/숫자가 아닌 것은 모두 공백으로 (한글 버림)
    text = re.sub(r"[^0-9A-Za-z]+", " ", text)
    text = text.strip()
    if not text:
        return ""

    # 소문자화 + stop word 제거
    tokens: list[str] = []
    for tok in text.split():
        tok_lower = tok.lower()
        if not re.fullmatch(r"[a-z0-9]+", tok_lower):
            continue
        if tok_lower in _STOP_WORDS:
            continue
        tokens.append(tok_lower)

    # 최대 max_words 토큰
    tokens = tokens[:max_words]
    return "_".join(tokens)


# ────────────────────────────────────────────────────────────
# 종합 생성기
# ────────────────────────────────────────────────────────────

def make_concept_id(
    module_id: str,
    concept_name: str,
    *,
    used_ids: Optional[Iterable[str]] = None,
    max_slug_words: int = 4,
) -> str:
    """module_id + concept_name 으로 concept_id 생성, 충돌 시 _v2/_v3 부여.

    >>> make_concept_id("M1_4", "Taylor's 4 Principles")
    'M1_4_taylors_4_principles'
    >>> make_concept_id("M2_1_5", "KJ Method")
    'M2_1_5_kj_method'
    >>> make_concept_id("manufacturing_overview", "What is manufacturing")
    'manufacturing_overview_what_manufacturing'

    충돌 회피:
    >>> existing = {"M1_4_taylor_principles"}
    >>> make_concept_id("M1_4", "Taylor Principles", used_ids=existing)
    'M1_4_taylor_principles_v2'
    """
    slug = slugify(concept_name, max_words=max_slug_words)
    if not slug:
        # concept_name 이 비정상일 때의 최후 fallback
        slug = "concept"

    base = f"{module_id}_{slug}"
    if used_ids is None:
        return base

    used_set = set(used_ids)
    if base not in used_set:
        return base
    # v2, v3, ... 추가
    for i in range(2, 100):
        candidate = f"{base}_v{i}"
        if candidate not in used_set:
            return candidate
    # 극단적 충돌 (실용상 도달 불가)
    raise RuntimeError(f"concept_id 충돌 회피 실패: base={base}")


if __name__ == "__main__":
    import doctest

    failed, total = doctest.testmod(verbose=True)
    print(f"\n{total - failed}/{total} doctests passed")
