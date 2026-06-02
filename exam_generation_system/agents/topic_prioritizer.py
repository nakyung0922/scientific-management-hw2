"""
Topic Prioritizer — Agent 1-4 (PG1, R3).

입력: ConceptKnowledgeStructure + embeddings dict + ReqVector
출력: ConceptKnowledgeStructure (priority_score 필드 채워짐)

설계 원칙:
- LLM 호출 없음. Python 결정론 계산만 (cosine similarity + 가중합).
- 입력 구조의 노드를 in-place 수정하지 않고 새 객체 반환 (불변성).
- 옵션 B 매칭: target이 모듈 ID면 prefix 매칭, concept_id면 정확 매칭.
- embeddings dict가 비면 graceful degradation — importance + intrinsic만으로 산출.

priority_score 산출 공식 (v0.1):
  score = 0.5·target_match_weight + 0.3·sim_weight + 0.2·importance_weight

  - target_match_weight: ReqVector.target_chapters_or_concepts에 매칭 시 1.0, 아니면 0.0
  - sim_weight: 매칭된 target들과의 평균 cosine similarity (없으면 0.0)
  - importance_weight: importance enum → 1.0/0.66/0.33

옵션 B 매칭 규칙:
  - target이 모듈 ID(예: "M1_1"): concept_id.startswith("M1_1_") 또는 concept_id == "M1_1"
  - target이 concept_id(예: "M1_4_taylor_principles"): concept_id == "M1_4_taylor_principles"
  - target이 모듈 root와 일치하면 그 모듈의 모든 노드가 매칭됨

LOA = 7. LLM 안 쓰지만 결정 규칙 자동화로 사람 개입 최소.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from common.concept_utils import cosine_similarity
from common.enums import Importance
from common.schemas import (
    ConceptKnowledgeStructure,
    ConceptNode,
    ReqVector,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "Topic_Prioritizer"
ROOT_CONCEPT_ID = "ROOT_DOCUMENT"

# 가중치 (합 = 1.0)
W_TARGET_MATCH = 0.5
W_SIMILARITY = 0.3
W_IMPORTANCE = 0.2

# Importance enum → 가중치 매핑
IMPORTANCE_WEIGHTS = {
    Importance.HIGH: 1.0,
    Importance.MEDIUM: 0.66,
    Importance.LOW: 0.33,
}


# ────────────────────────────────────────────────────────────
# 옵션 B 매칭 함수
# ────────────────────────────────────────────────────────────
def target_matches_concept(target: str, concept_id: str) -> bool:
    """target이 concept_id에 매칭되는지 (옵션 B).

    >>> target_matches_concept("M1_1", "M1_1_taylor_principles")
    True
    >>> target_matches_concept("M1_1", "M1_1")
    True
    >>> target_matches_concept("M1_1_taylor", "M1_1_taylor")
    True
    >>> target_matches_concept("M1_1_taylor", "M1_1_pig_iron")
    False
    >>> target_matches_concept("M1_1", "M1_10_other")
    False
    >>> target_matches_concept("M2_1", "M2_1_2_kj_method")
    True
    """
    if target == concept_id:
        return True
    # prefix 매칭: target이 모듈 ID이고 concept가 그 하위
    # "M1_1"이 "M1_10"에 매칭되면 안 되므로 _로 끝나는 prefix 검사
    return concept_id.startswith(target + "_")


def find_matching_concepts(
    target: str,
    structure: ConceptKnowledgeStructure,
) -> list[str]:
    """주어진 target에 매칭되는 모든 concept_id 반환 (ROOT 제외)."""
    return [
        n.concept_id
        for n in structure.nodes
        if n.concept_id != ROOT_CONCEPT_ID
        and target_matches_concept(target, n.concept_id)
    ]


# ────────────────────────────────────────────────────────────
# Target embedding 추정
# ────────────────────────────────────────────────────────────
def compute_target_embeddings(
    targets: list[str],
    embeddings: dict[str, list[float]],
    structure: ConceptKnowledgeStructure,
) -> dict[str, list[float]]:
    """각 target의 representative embedding을 계산.

    - target이 단일 concept과 매칭되면 그 embedding 사용
    - target이 여러 concept과 매칭되면 평균 (centroid)
    - 매칭되는 concept이 embeddings에 없으면 target은 skip

    Returns:
        {target: 768d centroid vector}
    """
    result: dict[str, list[float]] = {}
    for target in targets:
        matched = find_matching_concepts(target, structure)
        matched_vecs = [
            embeddings[cid] for cid in matched if cid in embeddings
        ]
        if not matched_vecs:
            continue
        # centroid
        dim = len(matched_vecs[0])
        centroid = [
            sum(v[i] for v in matched_vecs) / len(matched_vecs)
            for i in range(dim)
        ]
        result[target] = centroid
    return result


# ────────────────────────────────────────────────────────────
# 메인 Prioritizer 클래스
# ────────────────────────────────────────────────────────────
class TopicPrioritizer:
    """Topic Prioritizer Agent (LLM 없음, 결정론 계산).

    Usage:
        prioritizer = TopicPrioritizer()
        enriched = prioritizer.prioritize(
            structure, embeddings, req_vector
        )
    """

    def prioritize(
        self,
        structure: ConceptKnowledgeStructure,
        embeddings: dict[str, list[float]],
        req_vector: ReqVector,
    ) -> ConceptKnowledgeStructure:
        """각 노드의 priority_score를 채워서 새 ConceptKnowledgeStructure 반환.

        원본 structure는 수정하지 않음 (불변성).
        ROOT_DOCUMENT는 priority_score=0.0으로 둠 (ExamPlanner가 어차피 필터링).
        """
        targets = req_vector.target_chapters_or_concepts or []

        # 각 target의 centroid embedding
        target_embeddings = compute_target_embeddings(
            targets, embeddings, structure
        )

        # 각 노드별로 score 계산
        new_nodes: list[ConceptNode] = []
        for node in structure.nodes:
            if node.concept_id == ROOT_CONCEPT_ID:
                # ROOT는 점수 없음
                new_nodes.append(node.model_copy(update={"priority_score": 0.0}))
                continue

            score = self._compute_node_score(
                node, embeddings, target_embeddings, targets,
            )
            new_nodes.append(node.model_copy(update={"priority_score": score}))

        return ConceptKnowledgeStructure(
            nodes=new_nodes,
            edges=list(structure.edges),  # edges는 그대로
        )

    # ── 점수 계산 ────────────────────────────────────────────
    def _compute_node_score(
        self,
        node: ConceptNode,
        embeddings: dict[str, list[float]],
        target_embeddings: dict[str, list[float]],
        targets: list[str],
    ) -> float:
        # 1. target match weight (0 또는 1)
        target_match = self._target_match_weight(node, targets)

        # 2. similarity weight (0~1)
        sim = self._similarity_weight(node, embeddings, target_embeddings)

        # 3. importance weight (0.33~1)
        imp = IMPORTANCE_WEIGHTS.get(node.importance, 0.5)

        # priority_score = 0.5·target_match + 0.3·cosine_sim + 0.2·importance
        # target_match: ReqVector 지정 챕터/개념에 속하면 1.0, 아니면 0.0
        # cosine_sim:   노드 embedding과 target centroid 간 평균 유사도 (→ [0,1])
        # importance:   high=1.0 / medium=0.66 / low=0.33
        score = (
            W_TARGET_MATCH * target_match
            + W_SIMILARITY * sim
            + W_IMPORTANCE * imp
        )
        # [0, 1] 클램프 (가중치 합이 1이고 각 항이 [0,1]이라 클램프는 안전장치)
        return max(0.0, min(1.0, score))

    def _target_match_weight(
        self,
        node: ConceptNode,
        targets: list[str],
    ) -> float:
        """노드가 어떤 target에라도 매칭되면 1.0, 아니면 0.0."""
        for target in targets:
            if target_matches_concept(target, node.concept_id):
                return 1.0
        return 0.0

    def _similarity_weight(
        self,
        node: ConceptNode,
        embeddings: dict[str, list[float]],
        target_embeddings: dict[str, list[float]],
    ) -> float:
        """노드 embedding과 모든 target centroid의 평균 cosine similarity.

        cosine similarity는 [-1, 1] 범위이므로 (x+1)/2로 [0, 1]에 맵핑.
        embedding이 없거나 target_embeddings가 비면 0.5 (중립).
        """
        node_vec = embeddings.get(node.concept_id)
        if not node_vec or not target_embeddings:
            return 0.5  # 중립값 (정보 없을 때 페널티 X)

        sims = []
        for tvec in target_embeddings.values():
            cos = cosine_similarity(node_vec, tvec)
            # [-1, 1] → [0, 1]
            sims.append((cos + 1.0) / 2.0)

        if not sims:
            return 0.5
        return sum(sims) / len(sims)


# ────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import doctest
    failed, total = doctest.testmod(verbose=False)
    print(f"✓ doctests: {total - failed}/{total} passed")

    # 옵션 B 매칭 추가 case
    cases = [
        ("M1_1", "M1_1_taylor", True),
        ("M1_1", "M1_1", True),
        ("M1_1", "M1_10_other", False),   # prefix 함정
        ("M2_1_2", "M2_1_2_kj_method", True),
        ("manufacturing_overview", "manufacturing_overview", True),
        ("M1", "M1_1_taylor", True),       # 모듈 단위 큰 prefix
    ]
    for target, cid, expected in cases:
        actual = target_matches_concept(target, cid)
        mark = "✓" if actual == expected else "✗"
        print(f"  {mark} match({target!r}, {cid!r}) = {actual} (expected {expected})")
