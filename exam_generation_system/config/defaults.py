"""
Central configuration for Agentic Exam Generation System.

v0.4 스키마의 config_defaults를 Python으로 옮긴 모듈.
회의 결과나 실험을 통해 수치를 조정하려면 이 파일만 수정하면 됨.

환경변수가 설정된 경우 일부 값은 그것으로 override됨.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# .env 파일 로드 (있으면)
load_dotenv()

# ────────────────────────────────────────────────────────────
# 프로젝트 경로
# ────────────────────────────────────────────────────────────
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
PROMPTS_DIR: Final[Path] = PROJECT_ROOT / "config" / "prompts"


# ────────────────────────────────────────────────────────────
# 시험 기본 구성 (v0.5 결정사항 반영: 75분, 10문항, 100점)
# ────────────────────────────────────────────────────────────
EXAM_BASICS: Final[dict] = {
    "exam_type": "midterm",
    "total_points": 100,
    "duration_minutes": 75,
    "evaluation_focus": "balanced",  # balanced | knowledge_focused | application_focused
}


# ────────────────────────────────────────────────────────────
# 문항 유형 분포 (v0.5: 논술형 중심)
# ────────────────────────────────────────────────────────────
QUESTION_COUNT: Final[dict] = {
    "short_answer": 2,
    "long_answer": 5,
    "case_analysis": 3,
    # MCQ_single은 실제 중간고사에 거의 없다는 정보 반영 — 제외
    "total": 10,
}

POINTS_PER_TYPE: Final[dict] = {
    "short_answer": 10,
    "long_answer": 12,
    "case_analysis": 7,
}
# 합산 시 101점이 나오므로 Exam Planner가 case_analysis 슬롯 끝에서 1점 차감하여 100점으로 맞춤


# ────────────────────────────────────────────────────────────
# 난이도 분포 (그룹 단위)
# ────────────────────────────────────────────────────────────
DIFFICULTY_DISTRIBUTION_GROUP: Final[dict] = {
    "easy": 2,   # → L1 1, L2 1
    "medium": 6, # → L3 6
    "hard": 2,   # → L4 1, L5 1
}

BASELINE_LEVEL: Final[int] = 3  # L3 = medium baseline


# ────────────────────────────────────────────────────────────
# 난이도 산출 공식 가중치
# ────────────────────────────────────────────────────────────
EVALUATION_FOCUS_WEIGHTS: Final[dict] = {
    "balanced":            {"alpha": 0.5, "beta": 0.5},
    "knowledge_focused":   {"alpha": 0.7, "beta": 0.3},
    "application_focused": {"alpha": 0.3, "beta": 0.7},
}

# X₁ 계산용: 그래프 depth를 0~1로 정규화
X1_DEPTH_MAPPING: Final[dict] = {
    0: 0.00,
    1: 0.25,
    2: 0.50,
    3: 0.75,
    4: 1.00,  # 4 이상은 모두 1.00
}

# X₂ 계산용: 문항 유형별 인지 부하 (Bloom's Taxonomy 기반)
X2_QUESTION_TYPE_MAPPING: Final[dict] = {
    "MCQ_single":    0.2,  # 사용하지 않지만 enum 호환 위해 정의
    "short_answer":  0.4,
    "long_answer":   0.7,
    "case_analysis": 1.0,
}

# D 값을 1~5 레벨로 매핑하는 임계값
LEVEL_THRESHOLDS: Final[dict] = {
    # (lower_inclusive, upper_exclusive) — L5만 inclusive
    1: (0.00, 0.20),
    2: (0.20, 0.40),
    3: (0.40, 0.60),
    4: (0.60, 0.80),
    5: (0.80, 1.01),  # 1.0 inclusive
}


# ────────────────────────────────────────────────────────────
# 유형-레벨 제약
# ────────────────────────────────────────────────────────────
QUESTION_TYPE_LEVEL_RANGE: Final[dict] = {
    "short_answer":  (1, 4),  # L1~L4
    "long_answer":   (2, 5),  # L2~L5
    "case_analysis": (3, 5),  # L3~L5 (L1·L2 불가)
    "MCQ_single":    (1, 2),  # 사용 안 함, 호환용
}


# ────────────────────────────────────────────────────────────
# Tester 임계값
# ────────────────────────────────────────────────────────────
THRESHOLDS: Final[dict] = {
    "factfulness_pass":           0.80,
    "difficulty_alignment_pass":  0.60,
    "max_question_rework_attempts": 3,
}


# ────────────────────────────────────────────────────────────
# Paper Formattor
# ────────────────────────────────────────────────────────────
OUTPUT_FORMAT: Final[str] = "both"  # docx + pdf

# 시험지 헤더 표준
EXAM_HEADER_TEMPLATE: Final[dict] = {
    "course_name": "Scientific Management",
    "professor": "Woojin Park",
    "exam_label": "중간고사 (Midterm)",
    "fields": ["학번", "이름"],
}


# ────────────────────────────────────────────────────────────
# 시험 범위 (M1~M3 정리본 기준 + 비디오 강의)
# ────────────────────────────────────────────────────────────
TARGET_CHAPTERS: Final[list[str]] = [
    "M1_1", "M1_2", "M1_3", "M1_4", "M1_5",
    "M2_1_1", "M2_1_2", "M2_1_3", "M2_1_5",
    "M3_1_1", "manufacturing_overview",
]

EXAM_SCOPE_MODULES: Final[list[str]] = TARGET_CHAPTERS


# ────────────────────────────────────────────────────────────
# Paper Formattor 메타데이터
# ────────────────────────────────────────────────────────────
EXAM_METADATA: Final[dict] = {
    "course_name_kr": "과학적 관리",
    "course_name_en": "Scientific Management",
    "semester": "2026-1",
    "total_points": 100,
    "duration_minutes": 75,
    "total_questions": 10,
}

EXAM_INSTRUCTIONS_KR: Final[list[str]] = [
    "시험 시간은 75분이며, 총 10문항 100점 만점입니다.",
    "답안은 제공된 답안 공간에 명확히 작성하십시오.",
    "단답형 2문항, 서술형 5문항, 사례 분석 3문항으로 구성됩니다.",
    "필기구는 검정 또는 파란색 펜만 사용 가능합니다.",
]

OUTPUT_DIR_TEMPLATE: Final[str] = "outputs/{session_id}"


# ────────────────────────────────────────────────────────────
# LLM 설정 (Gemini Vertex AI)
# ────────────────────────────────────────────────────────────
LLM_CONFIG: Final[dict] = {
    "provider": "vertex_ai",
    "project_id": os.getenv("GCP_PROJECT_ID", ""),
    "region": os.getenv("GCP_REGION", "asia-northeast3"),
    "models": {
        "heavy": os.getenv("GEMINI_MODEL_HEAVY", "gemini-2.5-pro"),
        "light": os.getenv("GEMINI_MODEL_LIGHT", "gemini-2.5-flash"),
    },
    # agent별 모델 매핑
    "agent_model_assignment": {
        "Exam_Planner":        "heavy",   # 복잡한 계획 + 그래프 추론
        "QA_Generator_heavy":  "heavy",   # case_analysis, long_answer
        "QA_Generator_light":  "light",   # short_answer
        "Factfulness_Tester":  "heavy",   # 정확성이 중요
        "Difficulty_Tester":   "heavy",   # 5단계 척도 판정
        "Rubric_Machine":      "light",   # 비교적 단순
        "Paper_Formattor":     "light",   # LLM 거의 안 씀
    },
}

# 디버그/개발 모드
DEBUG_MODE: Final[bool] = os.getenv("DEBUG_MODE", "false").lower() == "true"
ENABLE_USAGE_LOGGING: Final[bool] = (
    os.getenv("ENABLE_USAGE_LOGGING", "true").lower() == "true"
)


# ────────────────────────────────────────────────────────────
# Sanity check
# ────────────────────────────────────────────────────────────
def _validate() -> None:
    """Module load 시점에 기본값들이 일관성 있는지 검증."""
    # 문항 수 합계
    assert (
        sum(v for k, v in QUESTION_COUNT.items() if k != "total")
        == QUESTION_COUNT["total"]
    ), "QUESTION_COUNT 합계 불일치"

    # 난이도 그룹 합계 == 총 문항 수
    assert (
        sum(DIFFICULTY_DISTRIBUTION_GROUP.values()) == QUESTION_COUNT["total"]
    ), "DIFFICULTY_DISTRIBUTION_GROUP 합계 != 총 문항 수"

    # evaluation_focus가 weights에 정의되어 있어야 함
    assert (
        EXAM_BASICS["evaluation_focus"] in EVALUATION_FOCUS_WEIGHTS
    ), f"evaluation_focus '{EXAM_BASICS['evaluation_focus']}' 미정의"

    # 레벨 임계값 연속성
    for level in range(1, 5):
        assert (
            LEVEL_THRESHOLDS[level][1] == LEVEL_THRESHOLDS[level + 1][0]
        ), f"LEVEL_THRESHOLDS L{level}와 L{level+1} 사이 gap/overlap"


_validate()


if __name__ == "__main__":
    # 직접 실행하면 현재 config를 출력
    import json
    config_snapshot = {
        "exam_basics": EXAM_BASICS,
        "question_count": QUESTION_COUNT,
        "points_per_type": POINTS_PER_TYPE,
        "difficulty_distribution_group": DIFFICULTY_DISTRIBUTION_GROUP,
        "thresholds": THRESHOLDS,
        "target_chapters": TARGET_CHAPTERS,
        "llm_models": LLM_CONFIG["models"],
    }
    print(json.dumps(config_snapshot, ensure_ascii=False, indent=2))
