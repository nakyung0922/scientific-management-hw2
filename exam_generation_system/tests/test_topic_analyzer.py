"""
Unit tests for Topic_Analyzer + EmbeddingClient.

Mock LLM + Mock Embedding 모드.
R4와의 계약 (9개 invariants)을 모두 검증.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.topic_analyzer import (
    ROOT_CONCEPT_ID,
    TopicAnalyzer,
    sanitize_slug,
)
from common.embedding_client import EMBEDDING_DIM, EmbeddingClient, _hash_based_mock
from common.enums import DifficultyLevel, EdgeRelation, Importance, QuestionType
from common.gemini_client import GeminiClient, _tracker, reset_usage
from common.schemas import (
    ConceptKnowledgeStructure,
    ConceptSummary,
)


# ────────────────────────────────────────────────────────────
# Fixtures (test inputs)
# ────────────────────────────────────────────────────────────
def _summary(module_id: str, title: str, phrases: list[str]) -> ConceptSummary:
    return ConceptSummary(
        module_id=module_id,
        source_filename=f"{module_id}.pdf",
        title=title,
        summary_text=f"{title}에 관한 모듈. " + " ".join(phrases),
        key_phrases=phrases,
        page_count=10,
    )


def _make_responder(payload: dict):
    def responder(system: str, user: str) -> str:
        return json.dumps(payload, ensure_ascii=False)
    return responder


def _good_responder():
    """3개 모듈, 5개 노드, related edge 1개."""
    return _make_responder({
        "nodes": [
            {
                "module_id": "M1_4",
                "concept_name": "Taylor의 과학적 관리 4원칙",
                "slug": "taylor_principles",
                "parent_id": "ROOT_DOCUMENT",
                "B_score": 3,
                "suitable_question_types": ["long_answer", "case_analysis"],
                "importance": "high",
                "source_pages": ["M1.4 p.3"],
            },
            {
                "module_id": "M1_4",
                "concept_name": "Pig Iron Case",
                "slug": "pig_iron_case",
                "parent_id": "M1_4_taylor_principles",
                "B_score": 2,
                "suitable_question_types": ["short_answer", "case_analysis"],
                "importance": "medium",
                "source_pages": ["M1.4 p.4"],
            },
            {
                "module_id": "M2_1_1",
                "concept_name": "DASSI 5단계",
                "slug": "dassi",
                "parent_id": "ROOT_DOCUMENT",
                "B_score": 3,
                "suitable_question_types": ["long_answer", "case_analysis"],
                "importance": "high",
                "source_pages": ["M2.1.1 p.2"],
            },
            {
                "module_id": "M2_1_2",
                "concept_name": "KJ Method",
                "slug": "kj_method",
                "parent_id": "ROOT_DOCUMENT",
                "B_score": 3,
                "suitable_question_types": ["long_answer", "case_analysis"],
                "importance": "high",
                "source_pages": ["M2.1.2 p.2"],
            },
            {
                "module_id": "M3_1_1",
                "concept_name": "Therbligs",
                "slug": "therbligs",
                "parent_id": "ROOT_DOCUMENT",
                "B_score": 2,
                "suitable_question_types": ["short_answer", "long_answer"],
                "importance": "high",
                "source_pages": ["M3.1.1 p.2"],
            },
        ],
        "extra_edges": [
            {"from": "M2_1_1_dassi", "to": "M2_1_2_kj_method",
             "relation": "related", "weight": 0.8},
        ],
        "warnings": [],
    })


def _good_summaries() -> list[ConceptSummary]:
    return [
        _summary("M1_4", "Taylor's Scientific Management",
                 ["Taylor의 4원칙", "Pig Iron Case"]),
        _summary("M2_1_1", "DASSI", ["DASSI 5단계"]),
        _summary("M2_1_2", "KJ Method", ["KJ Method"]),
        _summary("M3_1_1", "Therbligs", ["Therbligs 17개"]),
    ]


def _new_analyzer(responder=None):
    """기본 분석기 생성 (mock 모드)."""
    if responder is None:
        responder = _good_responder()
    return TopicAnalyzer(
        client=GeminiClient(mock=True, mock_responder=responder),
        embedding_client=EmbeddingClient(mock=True),
    )


# ────────────────────────────────────────────────────────────
# 1. sanitize_slug 유닛
# ────────────────────────────────────────────────────────────
def test_sanitize_slug_basic():
    assert sanitize_slug("kj_method") == "kj_method"
    assert sanitize_slug("KJ_Method") == "kj_method"
    print("✓ test_sanitize_slug_basic")


def test_sanitize_slug_special_chars():
    assert sanitize_slug("Taylor's 4 Principles") == "taylor_s_4_principles"
    assert sanitize_slug("a__b___c") == "a_b_c"
    print("✓ test_sanitize_slug_special_chars")


def test_sanitize_slug_too_long():
    assert sanitize_slug("one_two_three_four_five") == "one_two_three_four"
    print("✓ test_sanitize_slug_too_long")


def test_sanitize_slug_korean_returns_empty():
    assert sanitize_slug("한글만") == ""
    print("✓ test_sanitize_slug_korean_returns_empty")


# ────────────────────────────────────────────────────────────
# 2. EmbeddingClient
# ────────────────────────────────────────────────────────────
def test_embedding_mock_dimension():
    client = EmbeddingClient(mock=True)
    vec = client.embed(agent_name="Test", text="hello")
    assert len(vec) == EMBEDDING_DIM
    print("✓ test_embedding_mock_dimension")


def test_embedding_mock_deterministic():
    client = EmbeddingClient(mock=True)
    v1 = client.embed(agent_name="Test", text="taylor")
    v2 = client.embed(agent_name="Test", text="taylor")
    assert v1 == v2
    print("✓ test_embedding_mock_deterministic")


def test_embedding_mock_distinct():
    client = EmbeddingClient(mock=True)
    v1 = client.embed(agent_name="Test", text="taylor")
    v2 = client.embed(agent_name="Test", text="kj method")
    assert v1 != v2
    print("✓ test_embedding_mock_distinct")


def test_embedding_logs_usage():
    reset_usage()
    client = EmbeddingClient(mock=True)
    client.embed(agent_name="Topic_Analyzer", text="x")
    records = [r for r in _tracker.records if r.agent_name == "Topic_Analyzer"]
    assert len(records) == 1
    assert "text-embedding-004" in records[0].model
    print("✓ test_embedding_logs_usage")


def test_embedding_custom_responder():
    """Custom responder가 호출되는지."""
    def fake(text):
        return [0.5] * EMBEDDING_DIM
    client = EmbeddingClient(mock=True, mock_responder=fake)
    vec = client.embed(agent_name="Test", text="anything")
    assert all(v == 0.5 for v in vec)
    print("✓ test_embedding_custom_responder")


# ────────────────────────────────────────────────────────────
# 3. analyze() — happy path
# ────────────────────────────────────────────────────────────
def test_analyze_happy_path_returns_tuple():
    analyzer = _new_analyzer()
    result = analyzer.analyze(_good_summaries(), session_id="abc")
    assert isinstance(result, tuple)
    assert len(result) == 2
    structure, embeddings = result
    assert isinstance(structure, ConceptKnowledgeStructure)
    assert isinstance(embeddings, dict)
    print("✓ test_analyze_happy_path_returns_tuple")


# ────────────────────────────────────────────────────────────
# 4. R4 계약 invariants (9개) - 가장 중요
# ────────────────────────────────────────────────────────────
def test_invariant_1_root_is_first_and_depth_zero():
    """invariant 1: nodes[0]은 ROOT_DOCUMENT, depth_in_tree=0."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    assert structure.nodes[0].concept_id == ROOT_CONCEPT_ID
    assert structure.nodes[0].depth_in_tree == 0
    print("✓ test_invariant_1_root_is_first_and_depth_zero")


def test_invariant_2_non_root_has_parent():
    """invariant 2: ROOT 제외 모든 노드의 parent_concept_id가 채워짐."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    for node in structure.nodes:
        if node.concept_id == ROOT_CONCEPT_ID:
            continue
        assert node.parent_concept_id is not None
        assert node.parent_concept_id != ""
    print("✓ test_invariant_2_non_root_has_parent")


def test_invariant_3_parent_id_exists_in_nodes():
    """invariant 3: parent_concept_id로 가리키는 노드가 실존."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    all_ids = {n.concept_id for n in structure.nodes}
    for node in structure.nodes:
        if node.parent_concept_id is None:
            continue
        assert node.parent_concept_id in all_ids, (
            f"dangling: {node.concept_id} → {node.parent_concept_id}"
        )
    print("✓ test_invariant_3_parent_id_exists_in_nodes")


def test_invariant_4_parent_of_has_child_of_pair():
    """invariant 4: 모든 parent_of 엣지가 child_of 쌍을 가짐."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    parent_pairs = {
        (e.from_concept_id, e.to_concept_id)
        for e in structure.edges
        if e.relation == EdgeRelation.PARENT_OF.value
    }
    child_pairs = {
        (e.from_concept_id, e.to_concept_id)
        for e in structure.edges
        if e.relation == EdgeRelation.CHILD_OF.value
    }
    # 각 parent_of(A→B)에 대해 child_of(B→A)가 있어야
    for a, b in parent_pairs:
        assert (b, a) in child_pairs, f"parent_of({a}→{b}) without child_of pair"
    print("✓ test_invariant_4_parent_of_has_child_of_pair")


def test_invariant_5_no_cycles():
    """invariant 5: 부모 체인에 사이클 없음."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    parent_map = {n.concept_id: n.parent_concept_id for n in structure.nodes}
    for start in parent_map:
        visited = set()
        current = start
        while current and current != ROOT_CONCEPT_ID:
            assert current not in visited, f"cycle at {current}"
            visited.add(current)
            current = parent_map.get(current)
    print("✓ test_invariant_5_no_cycles")


def test_invariant_6_concept_id_format():
    """invariant 6: concept_id에 공백·점 없음."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    for node in structure.nodes:
        assert " " not in node.concept_id
        assert "." not in node.concept_id
    print("✓ test_invariant_6_concept_id_format")


def test_invariant_7_suitable_types_not_empty():
    """invariant 7: ROOT 제외 모든 노드 suitable_question_types ≥ 1."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    for node in structure.nodes:
        if node.concept_id == ROOT_CONCEPT_ID:
            continue
        assert len(node.suitable_question_types) >= 1, (
            f"{node.concept_id} has empty suitable_question_types"
        )
    print("✓ test_invariant_7_suitable_types_not_empty")


def test_invariant_8_intrinsic_level_range():
    """invariant 8: intrinsic_difficulty_level이 1~5."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    for node in structure.nodes:
        level = int(node.intrinsic_difficulty_level)
        assert 1 <= level <= 5
    print("✓ test_invariant_8_intrinsic_level_range")


def test_invariant_9_embeddings_keys_subset_of_non_root():
    """invariant 9: embeddings_dict 키 ⊆ non-root concept_ids."""
    structure, embeddings = _new_analyzer().analyze(_good_summaries(), "abc")
    non_root_ids = {n.concept_id for n in structure.nodes
                    if n.concept_id != ROOT_CONCEPT_ID}
    assert set(embeddings.keys()).issubset(non_root_ids)
    # 모든 non-root 노드는 embedding을 가져야 (mock 클라이언트라 실패 없음)
    assert set(embeddings.keys()) == non_root_ids
    # 차원 확인
    for cid, vec in embeddings.items():
        assert len(vec) == EMBEDDING_DIM
    print("✓ test_invariant_9_embeddings_keys_subset_of_non_root")


# ────────────────────────────────────────────────────────────
# 5. Pass 2 결정론 — 그래프 계산 검증
# ────────────────────────────────────────────────────────────
def test_pass2_depth_calculation():
    """Pig Iron Case는 Taylor's principles의 자식 → depth=2."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    pig_iron = structure.get_node("M1_4_pig_iron_case")
    assert pig_iron is not None
    assert pig_iron.depth_in_tree == 2
    taylor = structure.get_node("M1_4_taylor_principles")
    assert taylor.depth_in_tree == 1
    print("✓ test_pass2_depth_calculation")


def test_pass2_intrinsic_level_formula():
    """level = round(0.4·A + 0.6·B)."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    # Taylor (B=3, parent=ROOT, has child) → A 일부 ↑, B=3 → level ~3
    taylor = structure.get_node("M1_4_taylor_principles")
    level = int(taylor.intrinsic_difficulty_level)
    assert 2 <= level <= 4  # B가 3이니 합리적 범위
    print("✓ test_pass2_intrinsic_level_formula")


def test_pass2_extra_edge_related():
    """related 엣지가 단방향 단일로 생성 (Rule C)."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    related = [e for e in structure.edges
               if e.relation == EdgeRelation.RELATED.value]
    assert len(related) == 1
    assert related[0].from_concept_id == "M2_1_1_dassi"
    assert related[0].to_concept_id == "M2_1_2_kj_method"
    print("✓ test_pass2_extra_edge_related")


def test_pass2_concept_id_assembly():
    """concept_id = {module_id}_{sanitized_slug}."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    ids = {n.concept_id for n in structure.nodes}
    assert "M1_4_taylor_principles" in ids
    assert "M2_1_2_kj_method" in ids
    print("✓ test_pass2_concept_id_assembly")


# ────────────────────────────────────────────────────────────
# 6. LLM 호출 횟수 (Pass 1 모듈 한꺼번에 1회)
# ────────────────────────────────────────────────────────────
def test_llm_called_once_for_all_modules():
    """전체 모듈을 한 번에 LLM에 던짐."""
    reset_usage()
    analyzer = _new_analyzer()
    analyzer.analyze(_good_summaries(), "abc")
    gemini_calls = [r for r in _tracker.records
                    if r.agent_name == "Topic_Analyzer"
                    and "embedding" not in r.model]
    assert len(gemini_calls) == 1
    print("✓ test_llm_called_once_for_all_modules")


def test_embedding_called_once_per_node():
    """각 non-root 노드마다 embedding 1회."""
    reset_usage()
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    embed_calls = [r for r in _tracker.records
                   if r.agent_name == "Topic_Analyzer"
                   and "embedding" in r.model]
    non_root_count = sum(1 for n in structure.nodes
                         if n.concept_id != ROOT_CONCEPT_ID)
    assert len(embed_calls) == non_root_count
    print("✓ test_embedding_called_once_per_node")


# ────────────────────────────────────────────────────────────
# 7. Fallback paths
# ────────────────────────────────────────────────────────────
def test_empty_input_returns_minimal_structure():
    analyzer = _new_analyzer()
    structure, embeddings = analyzer.analyze([], "abc")
    assert len(structure.nodes) == 1
    assert structure.nodes[0].concept_id == ROOT_CONCEPT_ID
    assert embeddings == {}
    print("✓ test_empty_input_returns_minimal_structure")


def test_llm_returns_invalid_json_uses_layer1_fallback():
    """LLM 실패 → 각 모듈당 overview 노드 1개씩."""
    def broken(system, user):
        return "garbage not json"
    analyzer = _new_analyzer(responder=broken)
    summaries = _good_summaries()
    structure, embeddings = analyzer.analyze(summaries, "abc")
    # ROOT + module당 1개 (4모듈)
    assert len(structure.nodes) == 1 + len(summaries)
    overview_ids = {f"{s.module_id}_overview" for s in summaries}
    actual_ids = {n.concept_id for n in structure.nodes
                  if n.concept_id != ROOT_CONCEPT_ID}
    assert actual_ids == overview_ids
    print("✓ test_llm_returns_invalid_json_uses_layer1_fallback")


def test_llm_returns_no_nodes_uses_layer1_fallback():
    """LLM이 nodes=[]만 줘도 Layer 1 fallback (Pass 2가 ValueError raise)."""
    responder = _make_responder({"nodes": [], "extra_edges": [], "warnings": []})
    analyzer = _new_analyzer(responder=responder)
    summaries = _good_summaries()
    structure, _ = analyzer.analyze(summaries, "abc")
    assert len(structure.nodes) == 1 + len(summaries)
    print("✓ test_llm_returns_no_nodes_uses_layer1_fallback")


# ────────────────────────────────────────────────────────────
# 8. 자체 정제 (slug, parent, cycle)
# ────────────────────────────────────────────────────────────
def test_invalid_slug_gets_fallback_name():
    """LLM이 한글 slug 줘도 정제되거나 fallback."""
    responder = _make_responder({
        "nodes": [{
            "module_id": "M1_1",
            "concept_name": "테스트",
            "slug": "한글만",  # sanitize 후 빈 문자열
            "parent_id": "ROOT_DOCUMENT",
            "B_score": 2,
            "suitable_question_types": ["short_answer", "long_answer"],
            "importance": "medium",
            "source_pages": [],
        }],
        "extra_edges": [],
        "warnings": [],
    })
    analyzer = _new_analyzer(responder=responder)
    structure, _ = analyzer.analyze(_good_summaries(), "abc")
    # concept_0 같은 fallback이 들어와야 함
    non_root = [n for n in structure.nodes if n.concept_id != ROOT_CONCEPT_ID]
    assert len(non_root) == 1
    assert "concept_" in non_root[0].concept_id
    print("✓ test_invalid_slug_gets_fallback_name")


def test_dangling_parent_id_forced_to_root():
    """LLM이 존재하지 않는 parent_id 주면 ROOT로 강제."""
    responder = _make_responder({
        "nodes": [{
            "module_id": "M1_1",
            "concept_name": "Test",
            "slug": "test_a",
            "parent_id": "M99_nonexistent",  # 존재 안 함
            "B_score": 2,
            "suitable_question_types": ["short_answer"],
            "importance": "medium",
            "source_pages": [],
        }],
        "extra_edges": [],
        "warnings": [],
    })
    analyzer = _new_analyzer(responder=responder)
    structure, _ = analyzer.analyze(_good_summaries(), "abc")
    node = structure.get_node("M1_1_test_a")
    assert node is not None
    assert node.parent_concept_id == ROOT_CONCEPT_ID
    print("✓ test_dangling_parent_id_forced_to_root")


def test_duplicate_slug_gets_v2_suffix():
    """같은 모듈에 같은 slug 두 번 → 두 번째는 _v2."""
    responder = _make_responder({
        "nodes": [
            {
                "module_id": "M1_1",
                "concept_name": "X1",
                "slug": "same",
                "parent_id": "ROOT_DOCUMENT",
                "B_score": 2,
                "suitable_question_types": ["short_answer"],
                "importance": "medium",
                "source_pages": [],
            },
            {
                "module_id": "M1_1",
                "concept_name": "X2",
                "slug": "same",
                "parent_id": "ROOT_DOCUMENT",
                "B_score": 2,
                "suitable_question_types": ["short_answer"],
                "importance": "medium",
                "source_pages": [],
            },
        ],
        "extra_edges": [],
        "warnings": [],
    })
    analyzer = _new_analyzer(responder=responder)
    structure, _ = analyzer.analyze(_good_summaries(), "abc")
    ids = {n.concept_id for n in structure.nodes}
    assert "M1_1_same" in ids
    assert "M1_1_same_v2" in ids
    print("✓ test_duplicate_slug_gets_v2_suffix")


def test_extra_edge_to_nonexistent_node_ignored():
    """extra_edge가 미존재 노드를 가리키면 무시."""
    responder = _make_responder({
        "nodes": [{
            "module_id": "M1_1",
            "concept_name": "X",
            "slug": "x",
            "parent_id": "ROOT_DOCUMENT",
            "B_score": 2,
            "suitable_question_types": ["short_answer"],
            "importance": "medium",
            "source_pages": [],
        }],
        "extra_edges": [
            {"from": "M1_1_x", "to": "M99_ghost",
             "relation": "related", "weight": 0.5},
        ],
        "warnings": [],
    })
    analyzer = _new_analyzer(responder=responder)
    structure, _ = analyzer.analyze(_good_summaries(), "abc")
    related = [e for e in structure.edges
               if e.relation == EdgeRelation.RELATED.value]
    assert related == []  # 무시됨
    print("✓ test_extra_edge_to_nonexistent_node_ignored")


# ────────────────────────────────────────────────────────────
# 9. 직렬화 호환성
# ────────────────────────────────────────────────────────────
def test_structure_serializable():
    """ConceptKnowledgeStructure가 JSON 직렬화 가능."""
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    js = structure.model_dump_json()
    rehydrated = ConceptKnowledgeStructure.model_validate_json(js)
    assert len(rehydrated.nodes) == len(structure.nodes)
    print("✓ test_structure_serializable")


# ────────────────────────────────────────────────────────────
# 10. PG2 helpers와 호환 (실제로 R4가 쓰는 메서드들)
# ────────────────────────────────────────────────────────────
def test_compat_get_node():
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    node = structure.get_node("M1_4_taylor_principles")
    assert node is not None
    assert node.concept_name == "Taylor의 과학적 관리 4원칙"
    print("✓ test_compat_get_node")


def test_compat_get_depth_between():
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    # Taylor와 Pig Iron Case는 부모-자식 → 거리 1
    d = structure.get_depth_between(
        "M1_4_taylor_principles", "M1_4_pig_iron_case"
    )
    assert d == 1
    print("✓ test_compat_get_depth_between")


def test_compat_get_assignable_nodes():
    """ROOT가 필터링되는지 (R4 ExamPlanner가 이걸 사용)."""
    from common.concept_utils import get_assignable_nodes
    structure, _ = _new_analyzer().analyze(_good_summaries(), "abc")
    assignable = get_assignable_nodes(structure)
    assert all(n.concept_id != ROOT_CONCEPT_ID for n in assignable)
    assert all(n.depth_in_tree != 0 for n in assignable)
    print("✓ test_compat_get_assignable_nodes")


# ────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_sanitize_slug_basic()
    test_sanitize_slug_special_chars()
    test_sanitize_slug_too_long()
    test_sanitize_slug_korean_returns_empty()
    test_embedding_mock_dimension()
    test_embedding_mock_deterministic()
    test_embedding_mock_distinct()
    print("\n✓ All Topic_Analyzer unit tests passed (subset)")
