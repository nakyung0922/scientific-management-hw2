"""
R6 작성 — HITL 인터페이스 강화 모듈
========================================

목적
----
Team 07 Supervisor의 _run_hitl() 함수가 MAX_RETRY를 초과한
(Factfulness/Difficulty 3회 연속 fail) 슬롯을 일반 warning과 동등하게
취급하여, 검토자가 미통과 슬롯의 존재와 위험성을 충분히 인지하지 못할
수 있다는 문제(보고서 §4.2 L2)를 해결하기 위한 R6 자체 패치 모듈.

설계 원리
---------
- 원본 supervisor.py 코드를 1줄도 수정하지 않음 (R5의 작업과 독립적)
- runtime monkey-patching으로 Supervisor 인스턴스의 _run_hitl 메서드만 교체
- 강화 버전은 MAX_RETRY 초과 슬롯을 별도 섹션으로 강조 표시하고,
  슬롯별 점수와 권장 행동을 명시함
- 패치 적용/해제가 자유로워 디버깅·비교 검증이 쉬움

사용 방법
---------
    from common.gemini_client import GeminiClient
    from agents.supervisor import Supervisor
    from r6_hitl_enhancer import patch_supervisor

    client = GeminiClient(mock=False)
    supervisor = Supervisor(client)

    # ⭐ 이 한 줄만 추가
    patch_supervisor(supervisor)

    result = supervisor.run(pdf_paths=[...], requirements="...")

해제 방법
---------
    supervisor 인스턴스를 새로 만들거나, 명시적으로 unpatch:
    from r6_hitl_enhancer import unpatch_supervisor
    unpatch_supervisor(supervisor)

작성자 : R6 (System Reviewer)
작성일 : 2026-05
관련 보고서 : R6_Critical_Discussion_Report.docx 부록 A
"""
from __future__ import annotations

import types
from typing import Optional

# Supervisor와 PipelineResult가 없는 환경에서도 모듈 import는 가능하도록 try/except
try:
    from agents.supervisor import Supervisor, PipelineResult
except ImportError:  # pragma: no cover
    Supervisor = None  # type: ignore
    PipelineResult = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────
# 원본 메서드 백업용 저장소 (unpatch를 위해 보존)
# ──────────────────────────────────────────────────────────────────────
_ORIGINAL_HITL_STORAGE: dict[int, callable] = {}


# ──────────────────────────────────────────────────────────────────────
# 강화된 HITL 함수
# ──────────────────────────────────────────────────────────────────────
def enhanced_hitl(self, result) -> tuple[bool, str]:
    """강화된 HITL — MAX_RETRY 초과 슬롯을 명시적으로 강조 표시.

    원본 _run_hitl()과의 차이점:
        1. 자동 검증 미통과 슬롯을 화면 상단에 별도 섹션으로 표시
        2. 슬롯별 factfulness/difficulty 점수와 시도 횟수 명시
        3. .docx 파일을 직접 확인하라는 명시적 안내
        4. 통계 요약 (총 문항, 통과율, 미통과 수)
        5. 미통과 슬롯이 있는 상태에서 승인 시 재경고
    """
    print("\n" + "=" * 70)
    print("👤  [HITL] Human Reviewer 검수 단계 (R6 강화 버전)")
    print("=" * 70)

    # ── 1. MAX_RETRY 초과 슬롯 강조 표시 ───────────────────────────
    exceeded_slots = [sr for sr in result.slot_results if sr.exceeded_max_retry]

    if exceeded_slots:
        print(f"\n  🚨  자동 검증 미통과 슬롯 {len(exceeded_slots)}건")
        print("  " + "─" * 66)
        print(f"  ⚠️   아래 슬롯은 Factfulness/Difficulty 3회 재시도 모두 실패했습니다.")
        print(f"  ⚠️   반드시 .docx 파일을 열어 해당 문항의 내용을 직접 확인하세요!")
        print()

        for sr in exceeded_slots:
            print(f"     ❌  슬롯 {sr.slot_id}")
            print(f"        - factfulness_score : {sr.factfulness_score:.2f}  (기준 ≥ 0.70)")
            print(f"        - difficulty_score  : {sr.difficulty_score:.2f}  (기준 ≥ 0.60)")
            print(f"        - attempts          : {sr.attempts}/3회 모두 실패")
            if sr.warnings:
                for w in sr.warnings:
                    print(f"        - 사유              : {w}")
            print()

        print("  " + "─" * 66)
        print(f"  💡  R6 권장: 위 {len(exceeded_slots)}건의 슬롯을 .docx에서 직접 확인 후 결정하세요.")
        print(f"     - 강의 범위와 일치하는지")
        print(f"     - 답안이 정확한지")
        print(f"     - 표현이 명확한지")
        print(f"     - 난이도가 적절한지")
        print("  " + "─" * 66)
    else:
        print("\n  ✅  모든 슬롯이 자동 검증을 통과했습니다.")

    # ── 2. 기타 warnings 표시 ──────────────────────────────────────
    other_warnings = [w for w in result.warnings
                      if "MAX_RETRY" not in w]
    if other_warnings:
        print(f"\n  ⚠️   기타 주의사항 {len(other_warnings)}건:")
        for w in other_warnings:
            print(f"     - {w}")

    # ── 3. 통계 요약 ───────────────────────────────────────────────
    total = len(result.slot_results)
    passed = total - len(exceeded_slots)
    pass_rate = (passed / total * 100) if total else 0

    print(f"\n  📊  통계 요약")
    print(f"     - 총 문항        : {total}개")
    print(f"     - 자동 검증 통과 : {passed}개 ({pass_rate:.1f}%)")
    print(f"     - 자동 검증 미통과: {len(exceeded_slots)}개")

    # ── 4. 산출물 경로 ─────────────────────────────────────────────
    if result.paper_output:
        print(f"\n  📄  시험지 : {result.paper_output.exam_paper_file}")
        print(f"  📄  답안지 : {result.paper_output.answer_key_file}")

    print("-" * 70)
    print("  💡  권장 행동")
    print("     1) 위에서 표시된 미통과 슬롯의 문항을 .docx에서 직접 확인")
    print("     2) 강의 범위 일치, 답안 정확성, 표현 명확성, 난이도 적절성 검토")
    print("     3) 문제 있으면 'n' 입력하여 반려 (현재는 전체 재실행 필요)")
    print("-" * 70)

    # ── 5. 사용자 입력 ────────────────────────────────────────────
    try:
        answer = input("\n  승인하시겠습니까? (y=승인 / n=반려): ").strip().lower()
    except EOFError:
        print("  [자동화 환경] EOFError → 자동 승인 처리")
        return True, ""

    # ── 6. 결정 후 메시지 ─────────────────────────────────────────
    if answer == "y":
        if exceeded_slots:
            print()
            print(f"  ⚠️   주의: 자동 검증 미통과 슬롯 {len(exceeded_slots)}건을 포함한 채 승인되었습니다.")
            print(f"  ⚠️   해당 문항이 실제 시험에 사용되기 전에 추가 검토를 권장합니다.")
        print("  ✅  Human Reviewer 승인 → 최종 출력 완료")
        return True, ""
    else:
        try:
            feedback = input("  반려 사유 입력 (선택): ").strip()
        except EOFError:
            feedback = ""
        print(f"  ↩️   반려 처리. 사유: {feedback or '없음'}")
        print(f"  ℹ️    참고: 현재 시스템은 부분 재생성을 지원하지 않습니다.")
        print(f"      supervisor.run()을 재실행하면 새 세션으로 처음부터 시작됩니다.")
        return False, feedback


# ──────────────────────────────────────────────────────────────────────
# 패치 적용/해제
# ──────────────────────────────────────────────────────────────────────
def patch_supervisor(supervisor) -> None:
    """Supervisor 인스턴스의 _run_hitl을 강화 버전으로 교체.

    원본 supervisor.py는 수정하지 않음. 인스턴스 단위 monkey-patching.

    Args:
        supervisor : Supervisor 인스턴스
    """
    key = id(supervisor)
    if key in _ORIGINAL_HITL_STORAGE:
        print("⚠️  [R6] 이미 패치가 적용되어 있습니다. 중복 적용 방지.")
        return

    # 원본 백업
    _ORIGINAL_HITL_STORAGE[key] = supervisor._run_hitl

    # 강화 버전으로 교체
    supervisor._run_hitl = types.MethodType(enhanced_hitl, supervisor)

    print("✔  [R6] HITL 인터페이스가 강화 버전으로 교체되었습니다.")
    print("    - MAX_RETRY 초과 슬롯이 강조 표시됩니다.")
    print("    - 통계 요약이 추가됩니다.")
    print("    - 검토 권장 행동이 안내됩니다.")


def unpatch_supervisor(supervisor) -> None:
    """패치 해제. 원본 _run_hitl로 복원.

    Args:
        supervisor : 패치가 적용된 Supervisor 인스턴스
    """
    key = id(supervisor)
    if key not in _ORIGINAL_HITL_STORAGE:
        print("⚠️  [R6] 해당 인스턴스는 패치가 적용되지 않았습니다.")
        return

    supervisor._run_hitl = _ORIGINAL_HITL_STORAGE.pop(key)
    print("✔  [R6] HITL이 원본으로 복원되었습니다.")


# ──────────────────────────────────────────────────────────────────────
# 단독 실행 시 자체 테스트
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """모듈 자체 점검 — Supervisor 없이도 import 가능한지 확인."""
    print("=" * 70)
    print("R6 HITL Enhancer — 자체 점검")
    print("=" * 70)
    print()
    print(f"  Supervisor import 가능 : {Supervisor is not None}")
    print(f"  patch_supervisor 정의   : {patch_supervisor is not None}")
    print(f"  unpatch_supervisor 정의 : {unpatch_supervisor is not None}")
    print(f"  enhanced_hitl 정의      : {enhanced_hitl is not None}")
    print()
    print("✔ 모듈이 정상적으로 로드되었습니다.")
    print()
    print("실제 사용 시:")
    print("    from r6_hitl_enhancer import patch_supervisor")
    print("    supervisor = Supervisor(client)")
    print("    patch_supervisor(supervisor)")
    print("    result = supervisor.run(...)")
