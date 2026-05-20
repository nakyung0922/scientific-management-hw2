"""
파일명 라벨링 + concept_id 생성 유틸.

Material Collector 는 시험 범위에 대해 어떤 가정도 하지 않는다.
어떤 PDF 가 들어오든 일관된 규칙으로:
  1) 파일명을 보고 module_id (이 PDF 한 권을 가리키는 안정적 라벨) 를 만든다
  2) LLM 이 추출한 concept_name 과 module_id 를 결합해
     v0.5 명세 형식의 concept_id 를 만든다 ({module_id}_{snake_case_slug})

v0.5 명세 (`agentic_system_schema_v0.5.json` concept_id_naming_rule):
    format   : {module_id}_{snake_case_slug}
    rules    :
      - 점(.) 은 언더스코어(_) 로 변환
      - 영문 소문자 + 숫자 + _ 만 허용 (module prefix 의 M 은 대문자 허용)
      - slug 최대 4 단어
      - 충돌 시 _v2 추가

라벨링 전략 (label_module_from_filename):
  - "M1.4.pdf"                          → "M1_4"        (M{숫자}.{숫자} 패턴 보존)
  - "M2.1.2_1.pdf"                      → "M2_1_2_1"
  - "KJ_Method.pdf"                     → "kj_method"   (영문 단어 합성)
  - "IntroductionOverviewManufacturing" → "introduction_overview_manufacturing"
                                                          (CamelCase 분해)
  - "자동화_개론.pdf"                    → "material_<hash>"
                                                          (ASCII 가 없으면 안정적 해시 라벨)
  - 충돌 시 _2, _3 부여
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional


# ────────────────────────────────────────────────────────────
# slug 화 (concept_name 등 자유 문자열용)
# ────────────────────────────────────────────────────────────

# 의미 없는 토큰
_STOP_WORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for",
    "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "as", "this", "that", "these", "those", "it", "its",
}


def slugify(text: str, max_words: int = 4) -> str:
    """자유 문자열을 v0.5 규칙의 snake_case slug 로.

    v0.5 명세: 영문 소문자 + 숫자 + _ 만 허용.
    한글/한자/일본어 등 비 ASCII 문자는 transliterate 가 비결정적이라
    여기서는 그냥 버린다.

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

    text = unicodedata.normalize("NFC", text)
    # 어퍼스트로피 류 제거
    text = text.replace("'", "").replace("\u2019", "").replace("\u2018", "")

    # ASCII 영문/숫자가 아닌 것은 모두 공백으로
    text = re.sub(r"[^0-9A-Za-z]+", " ", text).strip()
    if not text:
        return ""

    tokens: list[str] = []
    for tok in text.split():
        tok_lower = tok.lower()
        if not re.fullmatch(r"[a-z0-9]+", tok_lower):
            continue
        if tok_lower in _STOP_WORDS:
            continue
        tokens.append(tok_lower)

    return "_".join(tokens[:max_words])


# ────────────────────────────────────────────────────────────
# CamelCase 분해 보조
# ────────────────────────────────────────────────────────────

def _split_camel_case(text: str) -> str:
    """CamelCase 를 공백으로 분해.

    >>> _split_camel_case("IntroductionOverviewManufacturing")
    'Introduction Overview Manufacturing'
    >>> _split_camel_case("kjMethod")
    'kj Method'
    >>> _split_camel_case("XMLParser")
    'XML Parser'
    >>> _split_camel_case("already_snake")
    'already_snake'
    """
    # XMLParser → XML Parser  (대문자 연속 후 첫 대문자+소문자 경계)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    # introductionOverview → introduction Overview  (소문자→대문자 경계)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return text


# ────────────────────────────────────────────────────────────
# 파일명 → module_id 라벨링
# ────────────────────────────────────────────────────────────

# "M5_1", "M5.1", "M2.1.5", "M2.1.2_1" 같은 모듈 코드 패턴
_MODULE_CODE_PATTERN = re.compile(r"^M\d+(?:[._-]\d+)+", re.IGNORECASE)

# 파일명 stem 의 단일 토큰 자체가 모듈 코드 형태인지 (M\d 뒤 .\d+ 가 한 번 이상)
_PURE_MODULE_TOKEN = re.compile(r"^M\d+(?:[._-]\d+)+$", re.IGNORECASE)


def label_module_from_filename(
    filename: str,
    *,
    used_labels: Optional[Iterable[str]] = None,
    max_word_tokens: int = 8,
    fallback_hash_len: int = 6,
) -> str:
    """파일명을 보고 안정적인 module_id 를 생성한다.

    동작 우선순위:
      1) stem 이 'M5.1' 같은 모듈 코드 패턴으로 시작하면 점/하이픈을 '_' 로
         변환해 그대로 사용. (M 은 대문자 유지.)
      2) stem 에 ASCII 영문/숫자가 충분히 있으면 slugify + CamelCase 분해.
      3) ASCII 가 없으면 (예: 한글 파일명) `material_<hash>` 형태로 안정 라벨.

    used_labels 에 이미 있으면 _2, _3 ... 부여.

    >>> label_module_from_filename("M1.4.pdf")
    'M1_4'
    >>> label_module_from_filename("M2.1.2_1.pdf")
    'M2_1_2_1'
    >>> label_module_from_filename("M5_1_Automation.pdf")
    'M5_1_automation'
    >>> label_module_from_filename("IntroductionOverviewManufacturing.pdf")
    'introduction_overview_manufacturing'
    >>> label_module_from_filename("KJ Method.pdf")
    'kj_method'
    >>> label_module_from_filename("taylor's principles.pdf")
    'taylors_principles'

    충돌 회피:
    >>> label_module_from_filename("M1.4.pdf", used_labels={"M1_4"})
    'M1_4_2'
    """
    stem = Path(filename).stem

    label = _label_core(stem, max_word_tokens=max_word_tokens,
                        fallback_hash_len=fallback_hash_len, original=filename)

    if not used_labels:
        return label

    used_set = set(used_labels)
    if label not in used_set:
        return label
    for i in range(2, 1000):
        candidate = f"{label}_{i}"
        if candidate not in used_set:
            return candidate
    raise RuntimeError(f"module_id 충돌 회피 실패: {label}")


def _label_core(
    stem: str,
    *,
    max_word_tokens: int,
    fallback_hash_len: int,
    original: str,
) -> str:
    """핵심 라벨링 로직 (충돌 회피 제외)."""

    # 1) 'M\d+...' 모듈 코드 패턴이 stem 의 맨 앞에 있는지
    m = _MODULE_CODE_PATTERN.match(stem)
    if m:
        code = m.group(0)
        rest = stem[len(code):]
        # 코드 안의 점/하이픈 → _ ; M 은 대문자
        code_norm = "M" + re.sub(r"[.\-]", "_", code[1:])
        # 남은 부분이 의미 있으면 함께 붙이기
        rest_slug = slugify(rest, max_words=max_word_tokens) if rest else ""
        if rest_slug:
            # rest 가 단순한 구분자(_) 였다면 slug 가 빈문자가 됨
            return f"{code_norm}_{rest_slug}"
        return code_norm

    # 2) CamelCase 분해 후 slugify
    expanded = _split_camel_case(stem)
    slug = slugify(expanded, max_words=max_word_tokens)
    if slug:
        return slug

    # 3) ASCII 가 없음 (한글 파일명 등) → 안정적 해시 라벨
    h = hashlib.sha1(original.encode("utf-8")).hexdigest()[:fallback_hash_len]
    return f"material_{h}"


# ────────────────────────────────────────────────────────────
# concept_id 생성 (module_id + concept_name)
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
    >>> make_concept_id("introduction_overview_manufacturing", "What is manufacturing")
    'introduction_overview_manufacturing_what_manufacturing'

    충돌 회피:
    >>> existing = {"M1_4_taylor_principles"}
    >>> make_concept_id("M1_4", "Taylor Principles", used_ids=existing)
    'M1_4_taylor_principles_v2'
    """
    slug = slugify(concept_name, max_words=max_slug_words)
    if not slug:
        slug = "concept"

    base = f"{module_id}_{slug}"
    if used_ids is None:
        return base

    used_set = set(used_ids)
    if base not in used_set:
        return base
    for i in range(2, 100):
        candidate = f"{base}_v{i}"
        if candidate not in used_set:
            return candidate
    raise RuntimeError(f"concept_id 충돌 회피 실패: base={base}")


# ────────────────────────────────────────────────────────────
# 형식 검증 (validator)
# ────────────────────────────────────────────────────────────

_VALID_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def is_valid_id_format(s: str) -> bool:
    """v0.5 명세: 영문/숫자/_ 만 허용. (소문자 강제는 prefix M 예외 때문에 안 함)

    >>> is_valid_id_format("M1_4_taylor_principles")
    True
    >>> is_valid_id_format("introduction_overview_manufacturing")
    True
    >>> is_valid_id_format("has space")
    False
    >>> is_valid_id_format("has.dot")
    False
    >>> is_valid_id_format("")
    False
    """
    if not s:
        return False
    return bool(_VALID_ID_PATTERN.match(s))


if __name__ == "__main__":
    import doctest

    failed, total = doctest.testmod(verbose=True)
    print(f"\n{total - failed}/{total} doctests passed")
