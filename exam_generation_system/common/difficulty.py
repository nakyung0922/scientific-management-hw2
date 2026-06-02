"""
Difficulty computation: D = α·X₁ + β·X₂.

R1·R2가 합의한 정량 난이도 산출 모델.
- X₁: 개념 확장 범위 (graph depth/hop을 0~1로 정규화)
- X₂: 문항 유형 기반 인지 부하 (Bloom's Taxonomy)
- α, β: 평가 목표(evaluation_focus)에 따라 결정 (합 = 1.0)
- D를 L1~L5 정수로 매핑

이 모듈은 LLM을 호출하지 않으며, 순수 함수만 제공.
"""
from __future__ import annotations

from typing import Optional

from common.enums import (
    DifficultyLevel,
    EvaluationFocus,
    QuestionType,
)
from config.defaults import (
    EVALUATION_FOCUS_WEIGHTS,
    LEVEL_THRESHOLDS,
    QUESTION_TYPE_LEVEL_RANGE,
    X1_DEPTH_MAPPING,
    X2_QUESTION_TYPE_MAPPING,
)


# ────────────────────────────────────────────────────────────
# Weights
# ────────────────────────────────────────────────────────────
def get_weights(focus: EvaluationFocus) -> tuple[float, float]:
    """evaluation_focus에 대응하는 (α, β) 가중치 반환.

    >>> get_weights(EvaluationFocus.BALANCED)
    (0.5, 0.5)
    """
    weights = EVALUATION_FOCUS_WEIGHTS[focus.value]
    return weights["alpha"], weights["beta"]


# ────────────────────────────────────────────────────────────
# X₁: 개념 확장 범위
# ────────────────────────────────────────────────────────────
def compute_x1(depth_or_hop: int) -> float:
    """그래프 depth(또는 hop 거리)를 X₁ ∈ [0, 1]로 정규화.

    매핑 (config에 정의):
        0 → 0.00  (단일 노드)
        1 → 0.25  (인접 형제)
        2 → 0.50  (동일 부모 복수)
        3 → 0.75  (다른 범주 교차)
        ≥4 → 1.00 (광범위 통합)

    >>> compute_x1(0)
    0.0
    >>> compute_x1(2)
    0.5
    >>> compute_x1(10)
    1.0
    """
    if depth_or_hop < 0:
        raise ValueError(f"depth_or_hop must be non-negative, got {depth_or_hop}")
    clamped = min(depth_or_hop, 4)
    return X1_DEPTH_MAPPING[clamped]


# ────────────────────────────────────────────────────────────
# X₂: 문항 유형 인지 부하
# ────────────────────────────────────────────────────────────
def compute_x2(question_type: QuestionType) -> float:
    """문항 유형에 대응하는 X₂ ∈ [0, 1] 반환.

    매핑:
        MCQ_single   → 0.2
        short_answer → 0.4
        long_answer  → 0.7
        case_analysis → 1.0

    >>> compute_x2(QuestionType.LONG_ANSWER)
    0.7
    """
    return X2_QUESTION_TYPE_MAPPING[question_type.value]


# ────────────────────────────────────────────────────────────
# D: 종합 난이도
# ────────────────────────────────────────────────────────────
def compute_d(
    x1: float,
    x2: float,
    focus: EvaluationFocus = EvaluationFocus.BALANCED,
) -> float:
    """D = α·X₁ + β·X₂ 계산.

    >>> round(compute_d(0.5, 0.7, EvaluationFocus.BALANCED), 3)
    0.6
    >>> round(compute_d(0.5, 0.7, EvaluationFocus.APPLICATION_FOCUSED), 3)
    0.64
    """
    if not (0.0 <= x1 <= 1.0):
        raise ValueError(f"x1 must be in [0, 1], got {x1}")
    if not (0.0 <= x2 <= 1.0):
        raise ValueError(f"x2 must be in [0, 1], got {x2}")

    alpha, beta = get_weights(focus)
    return alpha * x1 + beta * x2


# ────────────────────────────────────────────────────────────
# D → Level 매핑
# ────────────────────────────────────────────────────────────
def d_to_level(d: float) -> DifficultyLevel:
    """D 값을 1~5 레벨로 매핑.

    임계값 (config 정의):
        [0.00, 0.20) → L1
        [0.20, 0.40) → L2
        [0.40, 0.60) → L3
        [0.60, 0.80) → L4
        [0.80, 1.00] → L5

    >>> d_to_level(0.0)
    <DifficultyLevel.L1: 1>
    >>> d_to_level(0.5)
    <DifficultyLevel.L3: 3>
    >>> d_to_level(1.0)
    <DifficultyLevel.L5: 5>
    """
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"d must be in [0, 1], got {d}")

    for level_int, (lower, upper) in LEVEL_THRESHOLDS.items():
        if lower <= d < upper:
            return DifficultyLevel(level_int)

    # d == 1.0 edge case (L5 upper bound는 1.01로 정의됨)
    return DifficultyLevel.L5


def level_to_d_range(level: DifficultyLevel) -> tuple[float, float]:
    """레벨에 대응하는 D 임계값 범위 (lower_inclusive, upper_exclusive) 반환.

    Difficulty_Tester가 verdict 판정 시 사용.
    """
    return LEVEL_THRESHOLDS[int(level)]


# ────────────────────────────────────────────────────────────
# 유형-레벨 제약
# ────────────────────────────────────────────────────────────
def is_valid_type_level_combo(
    question_type: QuestionType | str,
    level: DifficultyLevel | int,
) -> bool:
    """문항 유형과 난이도 레벨의 조합이 허용되는지 검증.

    제약:
        short_answer  → L1~L4
        long_answer   → L2~L5
        case_analysis → L3~L5

    >>> is_valid_type_level_combo(QuestionType.CASE_ANALYSIS, DifficultyLevel.L1)
    False
    >>> is_valid_type_level_combo(QuestionType.LONG_ANSWER, DifficultyLevel.L3)
    True
    >>> is_valid_type_level_combo("long_answer", 3)
    True
    """
    qtype_val = question_type.value if hasattr(question_type, "value") else question_type
    level_int = int(level)
    min_level, max_level = QUESTION_TYPE_LEVEL_RANGE[qtype_val]
    return min_level <= level_int <= max_level


def allowed_levels_for_type(question_type: QuestionType) -> list[DifficultyLevel]:
    """문항 유형이 만들 수 있는 레벨 목록.

    >>> allowed_levels_for_type(QuestionType.CASE_ANALYSIS)
    [<DifficultyLevel.L3: 3>, <DifficultyLevel.L4: 4>, <DifficultyLevel.L5: 5>]
    """
    min_level, max_level = QUESTION_TYPE_LEVEL_RANGE[question_type.value]
    return [DifficultyLevel(i) for i in range(min_level, max_level + 1)]


# ────────────────────────────────────────────────────────────
# 종합 계산 — Planner 작업의 핵심
# ────────────────────────────────────────────────────────────
def compute_slot_difficulty(
    depth_or_hop: int,
    question_type: QuestionType,
    focus: EvaluationFocus = EvaluationFocus.BALANCED,
) -> dict:
    """슬롯에 대한 X₁, X₂, D, level을 한 번에 계산.

    Exam Planner가 슬롯을 만들 때 expected_X1/X2/D/level을 채우는 데 사용.

    Returns:
        dict with keys: x1, x2, d, level, alpha, beta

    >>> result = compute_slot_difficulty(2, QuestionType.LONG_ANSWER)
    >>> result['x1'], result['x2']
    (0.5, 0.7)
    >>> round(result['d'], 3)
    0.6
    >>> result['level']
    <DifficultyLevel.L4: 4>
    """
    x1 = compute_x1(depth_or_hop)
    x2 = compute_x2(question_type)
    d = compute_d(x1, x2, focus)
    level = d_to_level(d)
    alpha, beta = get_weights(focus)

    return {
        "x1": x1,
        "x2": x2,
        "d": round(d, 4),
        "level": level,
        "alpha": alpha,
        "beta": beta,
    }


# ────────────────────────────────────────────────────────────
# 검증 헬퍼 — Difficulty_Tester가 사용
# ────────────────────────────────────────────────────────────
def difficulty_alignment_score(
    expected_d: float,
    estimated_d: float,
) -> float:
    """예상 D와 추정 D의 정렬도를 [0, 1]로 반환.

    score = 1 - |expected_D - estimated_D|
    1.0 = 완전 일치, 0.0 = 최대 불일치 (|diff| = 1.0).
    threshold = 0.60 미달 시 Exam_Planner에 difficulty_correction 요청.

    Difficulty_Tester가 verdict 판정에 사용.
    """
    diff = abs(expected_d - estimated_d)
    return max(0.0, 1.0 - diff)


if __name__ == "__main__":
    import doctest
    failed, total = doctest.testmod(verbose=True)
    print(f"\n{total - failed}/{total} doctests passed")
