"""
Unit tests for Topic_Prioritizer.

LLM 호출 없음. 순수 결정론 계산 검증.
"""
from __future__ import annotations

import pytest

from agents.topic_prioritizer import (
    IMPORTANCE_WEIGHTS,
    ROOT_CONCEPT_ID,
    TopicPrioritizer,
    W_IMPORTANCE,
    W_SIMILARITY,
    W_TARGET_MATCH,
    compute_target_embeddings,
    find_matching_concepts,
    target_matches_concept,
)
from common.embedding_client import EMBEDDING_DIM, _hash_based_mock
from common.enums import (
    DifficultyLevel,
    EdgeRelation,
    Importance,
    QuestionType,
)
from common.schemas import (
    ConceptEdge,
    ConceptKnowledgeStructure,
    ConceptNode,
    DifficultyDistributionGroup,
    QuestionTypeDistribution,
    ReqVector,
)
from common.enums import EvaluationFocus


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────
def _make_node(
    concept_id: str,
    *,
    importance: Importance = Importance.MEDIUM,
    intrinsic: DifficultyLevel = DifficultyLevel.L3,
    parent: str = ROOT_CONCEPT_ID,
    depth: int = 1,
) -> ConceptNode:
    return ConceptNode(
        concept_id=concept_id,
        concept_name=concept_id,
        depth_in_tree=depth,
        parent_concept_id=parent,
        importance=importance,
        intrinsic_difficulty_level=intrinsic,
        suitable_question_types=[QuestionType.LONG_ANSWER],
        source_pages=[],
    )


def _root_node() -> ConceptNode:
    return ConceptNode(
        concept_id=ROOT_CONCEPT_ID,
        concept_name="(root)",
        depth_in_tree=0,
        importance=Importance.LOW,
        intrinsic_difficulty_level=DifficultyLevel.L1,
        suitable_question_types=[],
    )


def _make_structure() -> ConceptKnowledgeStructure:
    return ConceptKnowledgeStructure(
        nodes=[
            _root_node(),
            _make_node("M1_1_work_definition", importance=Importance.HIGH),
            _make_node("M1_4_taylor_principles", importance=Importance.HIGH),
            _make_node("M1_4_pig_iron_case", importance=Importance.MEDIUM,
                       parent="M1_4_taylor_principles", depth=2),
            _make_node("M2_1_2_kj_method", importance=Importance.HIGH),
            _make_node("M3_1_1_therbligs", importance=Importance.LOW),
        ],
        edges=[
            ConceptEdge(from_concept_id=ROOT_CONCEPT_ID,
                        to_concept_id="M1_1_work_definition",
                        relation=EdgeRelation.PARENT_OF, weight=1.0),
        ],
    )


def _make_embeddings(
    structure: ConceptKnowledgeStructure,
) -> dict[str, list[float]]:
    """ROOT 제외 모든 노드에 mock embedding 부여."""
    return {
        n.concept_id: _hash_based_mock(n.concept_id)
        for n in structure.nodes
        if n.concept_id != ROOT_CONCEPT_ID
    }


def _make_req(targets: list[str]) -> ReqVector:
    return ReqVector(
        exam_title="Test",
        exam_type="midterm",
        target_chapters_or_concepts=targets,
        total_points=100,
        duration_minutes=75,
        evaluation_focus=EvaluationFocus.BALANCED,
        difficulty_distribution_group=DifficultyDistributionGroup(
            easy=2, medium=6, hard=2,
        ),
        question_type_distribution=QuestionTypeDistribution(
            MCQ_single=0, short_answer=2, long_answer=5, case_analysis=3,
        ),
        allowed_formats=["docx", "pdf"],
    )


# ────────────────────────────────────────────────────────────
# 1. target_matches_concept — 옵션 B 매칭 규칙
# ────────────────────────────────────────────────────────────
def test_match_exact_module_id():
    assert target_matches_concept("M1_1", "M1_1") is True
    print("✓ test_match_exact_module_id")


def test_match_module_prefix():
    assert target_matches_concept("M1_1", "M1_1_taylor_principles") is True
    print("✓ test_match_module_prefix")


def test_match_exact_concept_id():
    assert target_matches_concept(
        "M1_4_taylor_principles", "M1_4_taylor_principles"
    ) is True
    print("✓ test_match_exact_concept_id")


def test_match_different_module():
    assert target_matches_concept("M1_1", "M2_1_kj_method") is False
    print("✓ test_match_different_module")


def test_match_prefix_trap_M1_1_vs_M1_10():
    """핵심: 'M1_1'은 'M1_10_*'에 매칭되면 안 됨."""
    assert target_matches_concept("M1_1", "M1_10_other") is False
    assert target_matches_concept("M1_1", "M1_10") is False
    print("✓ test_match_prefix_trap_M1_1_vs_M1_10")


def test_match_module_prefix_M1():
    """광범위 prefix 'M1' → M1_*, M1_1_*, M1_2_* 모두 매칭."""
    assert target_matches_concept("M1", "M1_1_taylor") is True
    assert target_matches_concept("M1", "M1_4") is True
    assert target_matches_concept("M1", "M2_1") is False
    print("✓ test_match_module_prefix_M1")


def test_match_non_module_concept_id():
    assert target_matches_concept(
        "manufacturing_overview", "manufacturing_overview"
    ) is True
    assert target_matches_concept(
        "manufacturing", "manufacturing_overview"
    ) is True
    print("✓ test_match_non_module_concept_id")


# ────────────────────────────────────────────────────────────
# 2. find_matching_concepts
# ────────────────────────────────────────────────────────────
def test_find_module_matches_multiple_nodes():
    """target 'M1_4'는 M1_4_taylor, M1_4_pig_iron 둘 다 매칭."""
    structure = _make_structure()
    matches = find_matching_concepts("M1_4", structure)
    assert "M1_4_taylor_principles" in matches
    assert "M1_4_pig_iron_case" in matches
    assert "M1_1_work_definition" not in matches
    print("✓ test_find_module_matches_multiple_nodes")


def test_find_excludes_root():
    """ROOT는 어떤 target에도 매칭되지 않음."""
    structure = _make_structure()
    matches = find_matching_concepts("ROOT_DOCUMENT", structure)
    assert ROOT_CONCEPT_ID not in matches
    print("✓ test_find_excludes_root")


def test_find_empty_when_no_match():
    structure = _make_structure()
    matches = find_matching_concepts("M99_nonexistent", structure)
    assert matches == []
    print("✓ test_find_empty_when_no_match")


# ────────────────────────────────────────────────────────────
# 3. compute_target_embeddings
# ────────────────────────────────────────────────────────────
def test_target_embedding_single_match():
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    result = compute_target_embeddings(
        ["M1_4_taylor_principles"], embeddings, structure
    )
    assert "M1_4_taylor_principles" in result
    assert len(result["M1_4_taylor_principles"]) == EMBEDDING_DIM
    print("✓ test_target_embedding_single_match")


def test_target_embedding_centroid_for_module():
    """모듈 단위 target은 그 모듈 노드들의 centroid."""
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    result = compute_target_embeddings(["M1_4"], embeddings, structure)
    assert "M1_4" in result
    # centroid는 taylor와 pig_iron의 평균
    taylor = embeddings["M1_4_taylor_principles"]
    pig = embeddings["M1_4_pig_iron_case"]
    expected_first = (taylor[0] + pig[0]) / 2
    assert abs(result["M1_4"][0] - expected_first) < 1e-9
    print("✓ test_target_embedding_centroid_for_module")


def test_target_embedding_unmatched_skipped():
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    result = compute_target_embeddings(["M99_ghost"], embeddings, structure)
    assert result == {}
    print("✓ test_target_embedding_unmatched_skipped")


def test_target_embedding_empty_embeddings_skip():
    """embeddings dict이 비어있으면 모든 target skip."""
    structure = _make_structure()
    result = compute_target_embeddings(["M1_4"], {}, structure)
    assert result == {}
    print("✓ test_target_embedding_empty_embeddings_skip")


# ────────────────────────────────────────────────────────────
# 4. prioritize — 메인 인터페이스
# ────────────────────────────────────────────────────────────
def test_prioritize_returns_new_structure_not_mutated():
    """입력 structure는 수정되지 않음 (불변성)."""
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    req = _make_req(["M1_4"])

    original_scores = [n.priority_score for n in structure.nodes]

    result = TopicPrioritizer().prioritize(structure, embeddings, req)

    # 새 객체
    assert result is not structure
    # 원본은 변경 없음
    assert [n.priority_score for n in structure.nodes] == original_scores
    print("✓ test_prioritize_returns_new_structure_not_mutated")


def test_prioritize_all_nodes_have_score():
    """ROOT 포함 모든 노드의 priority_score가 채워짐."""
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    req = _make_req(["M1_4"])

    result = TopicPrioritizer().prioritize(structure, embeddings, req)

    for node in result.nodes:
        assert node.priority_score is not None
        assert 0.0 <= node.priority_score <= 1.0
    print("✓ test_prioritize_all_nodes_have_score")


def test_prioritize_root_has_zero_score():
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    req = _make_req(["M1_4"])

    result = TopicPrioritizer().prioritize(structure, embeddings, req)

    root = next(n for n in result.nodes if n.concept_id == ROOT_CONCEPT_ID)
    assert root.priority_score == 0.0
    print("✓ test_prioritize_root_has_zero_score")


def test_prioritize_matched_target_gets_higher_score():
    """target에 매칭된 노드가 매칭 안 된 노드보다 점수 높음."""
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    # M1_4만 target → M1_4_taylor, M1_4_pig_iron만 매칭
    req = _make_req(["M1_4"])

    result = TopicPrioritizer().prioritize(structure, embeddings, req)

    taylor = next(n for n in result.nodes
                  if n.concept_id == "M1_4_taylor_principles")
    therbligs = next(n for n in result.nodes
                     if n.concept_id == "M3_1_1_therbligs")

    assert taylor.priority_score > therbligs.priority_score
    print("✓ test_prioritize_matched_target_gets_higher_score")


def test_prioritize_importance_increases_score():
    """importance가 높을수록 점수 높음 (다른 조건 동일 시)."""
    # 동일 모듈에서 importance만 다른 두 노드
    structure = ConceptKnowledgeStructure(
        nodes=[
            _root_node(),
            _make_node("M9_1_high", importance=Importance.HIGH),
            _make_node("M9_1_low", importance=Importance.LOW),
        ],
        edges=[],
    )
    embeddings = {
        "M9_1_high": _hash_based_mock("M9_1_high"),
        "M9_1_low": _hash_based_mock("M9_1_low"),
    }
    req = _make_req([])  # target 없음 (importance만 차별 요소로)

    result = TopicPrioritizer().prioritize(structure, embeddings, req)
    high = next(n for n in result.nodes if n.concept_id == "M9_1_high")
    low = next(n for n in result.nodes if n.concept_id == "M9_1_low")
    assert high.priority_score > low.priority_score
    print("✓ test_prioritize_importance_increases_score")


def test_prioritize_empty_targets_uses_importance_only():
    """target 없으면 importance 차이로만 차별화."""
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    req = _make_req([])

    result = TopicPrioritizer().prioritize(structure, embeddings, req)

    # 모든 노드의 target_match는 0 → 점수 차이는 importance + similarity에서만
    high = next(n for n in result.nodes
                if n.concept_id == "M1_4_taylor_principles")  # HIGH
    low = next(n for n in result.nodes
               if n.concept_id == "M3_1_1_therbligs")  # LOW
    assert high.priority_score > low.priority_score
    print("✓ test_prioritize_empty_targets_uses_importance_only")


def test_prioritize_empty_embeddings_graceful():
    """embeddings dict이 비어도 동작 (degraded mode)."""
    structure = _make_structure()
    req = _make_req(["M1_4"])

    result = TopicPrioritizer().prioritize(structure, {}, req)

    for node in result.nodes:
        assert node.priority_score is not None
    # target 매칭만으로도 차별화 가능
    taylor = next(n for n in result.nodes
                  if n.concept_id == "M1_4_taylor_principles")
    therbligs = next(n for n in result.nodes
                     if n.concept_id == "M3_1_1_therbligs")
    assert taylor.priority_score > therbligs.priority_score
    print("✓ test_prioritize_empty_embeddings_graceful")


def test_prioritize_edges_preserved():
    """엣지는 그대로 보존."""
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    req = _make_req(["M1_4"])

    result = TopicPrioritizer().prioritize(structure, embeddings, req)
    assert len(result.edges) == len(structure.edges)
    print("✓ test_prioritize_edges_preserved")


def test_prioritize_node_count_preserved():
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    req = _make_req(["M1_4"])

    result = TopicPrioritizer().prioritize(structure, embeddings, req)
    assert len(result.nodes) == len(structure.nodes)
    print("✓ test_prioritize_node_count_preserved")


# ────────────────────────────────────────────────────────────
# 5. 점수 공식 검증
# ────────────────────────────────────────────────────────────
def test_score_formula_max_for_matched_high_importance():
    """target 매칭 + HIGH + 자기 자신과 sim=1 → 거의 max 점수."""
    structure = ConceptKnowledgeStructure(
        nodes=[
            _root_node(),
            _make_node("M1_1_focus", importance=Importance.HIGH),
        ],
        edges=[],
    )
    embeddings = {"M1_1_focus": _hash_based_mock("M1_1_focus")}
    req = _make_req(["M1_1_focus"])  # 자기 자신을 target으로

    result = TopicPrioritizer().prioritize(structure, embeddings, req)
    focus = next(n for n in result.nodes if n.concept_id == "M1_1_focus")

    # target_match=1, sim=1 (자기자신), importance=1 (HIGH)
    # score = 0.5·1 + 0.3·1 + 0.2·1 = 1.0
    assert focus.priority_score == pytest.approx(1.0, abs=0.01)
    print("✓ test_score_formula_max_for_matched_high_importance")


def test_score_weights_sum_to_one():
    """가중치 합이 1.0인지 확인 (정상 점수 범위 보장)."""
    assert W_TARGET_MATCH + W_SIMILARITY + W_IMPORTANCE == pytest.approx(1.0)
    print("✓ test_score_weights_sum_to_one")


def test_importance_weights_distinct():
    """3개 importance 값이 모두 다른 가중치."""
    weights = set(IMPORTANCE_WEIGHTS.values())
    assert len(weights) == 3
    print("✓ test_importance_weights_distinct")


# ────────────────────────────────────────────────────────────
# 6. 직렬화 호환성 (R4가 받을 형태)
# ────────────────────────────────────────────────────────────
def test_enriched_structure_serializable():
    """priority_score가 채워진 structure가 JSON 직렬화 가능."""
    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    req = _make_req(["M1_4"])

    result = TopicPrioritizer().prioritize(structure, embeddings, req)
    js = result.model_dump_json()
    rehydrated = ConceptKnowledgeStructure.model_validate_json(js)

    for orig, rehy in zip(result.nodes, rehydrated.nodes):
        assert orig.priority_score == rehy.priority_score
    print("✓ test_enriched_structure_serializable")


# ────────────────────────────────────────────────────────────
# 7. R4 계약 — Exam_Planner가 받을 객체로 깨끗
# ────────────────────────────────────────────────────────────
def test_compatible_with_exam_planner_interface():
    """ExamPlanner.plan()이 받는 형태(=ConceptKnowledgeStructure) 그대로."""
    from common.concept_utils import get_assignable_nodes

    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    req = _make_req(["M1_4"])

    result = TopicPrioritizer().prioritize(structure, embeddings, req)
    # ExamPlanner의 핵심 사용 패턴들이 동작하는지
    assignable = get_assignable_nodes(result)
    assert all(n.concept_id != ROOT_CONCEPT_ID for n in assignable)
    node = result.get_node("M1_4_taylor_principles")
    assert node is not None
    assert node.priority_score is not None
    print("✓ test_compatible_with_exam_planner_interface")


# ────────────────────────────────────────────────────────────
# 8. 의미 검증 — 우선순위 순위가 합리적
# ────────────────────────────────────────────────────────────
def test_priority_ranking_makes_sense():
    """매칭 + HIGH > 매칭 + MEDIUM > 비매칭 + HIGH > 비매칭 + LOW"""
    structure = ConceptKnowledgeStructure(
        nodes=[
            _root_node(),
            _make_node("M1_1_a", importance=Importance.HIGH),       # 매칭+HIGH
            _make_node("M1_1_b", importance=Importance.MEDIUM),     # 매칭+MEDIUM
            _make_node("M2_2_c", importance=Importance.HIGH),       # 비매칭+HIGH
            _make_node("M2_2_d", importance=Importance.LOW),        # 비매칭+LOW
        ],
        edges=[],
    )
    embeddings = {n.concept_id: _hash_based_mock(n.concept_id)
                  for n in structure.nodes if n.concept_id != ROOT_CONCEPT_ID}
    req = _make_req(["M1_1"])

    result = TopicPrioritizer().prioritize(structure, embeddings, req)
    scores = {n.concept_id: n.priority_score for n in result.nodes
              if n.concept_id != ROOT_CONCEPT_ID}

    assert scores["M1_1_a"] > scores["M1_1_b"]
    assert scores["M1_1_b"] > scores["M2_2_c"]  # 매칭의 큰 가중치(0.5) 덕분
    assert scores["M2_2_c"] > scores["M2_2_d"]
    print("✓ test_priority_ranking_makes_sense")


# ────────────────────────────────────────────────────────────
# 9. No LLM
# ────────────────────────────────────────────────────────────
def test_topic_prioritizer_makes_no_llm_calls():
    """LLM client 안 받음 + usage_tracker에 기록 안 됨."""
    from common.gemini_client import _tracker, reset_usage
    reset_usage()

    structure = _make_structure()
    embeddings = _make_embeddings(structure)
    req = _make_req(["M1_4"])

    TopicPrioritizer().prioritize(structure, embeddings, req)

    # Topic_Prioritizer 이름의 record가 0개여야 함
    records = [r for r in _tracker.records
               if r.agent_name == "Topic_Prioritizer"]
    assert len(records) == 0
    print("✓ test_topic_prioritizer_makes_no_llm_calls")


# ────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_match_exact_module_id()
    test_match_prefix_trap_M1_1_vs_M1_10()
    test_find_module_matches_multiple_nodes()
    test_prioritize_returns_new_structure_not_mutated()
    test_priority_ranking_makes_sense()
    print("\n✓ Topic_Prioritizer unit tests (subset) passed")
