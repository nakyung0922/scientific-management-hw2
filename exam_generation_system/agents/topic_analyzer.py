"""
Topic Analyzer — Agent 1-3 (PG1, R3).

입력: list[ConceptSummary] (Material_Collector 출력)
출력: tuple[ConceptKnowledgeStructure, dict[concept_id, list[float]]]
      (Exam_Planner 입력 + Topic_Prioritizer 입력)

3-Pass 구조:
  Pass 1 (LLM heavy, 1회): 노드/부모/B_score/유형/importance 추출 + extra_edges
  Pass 2 (Python 결정론):  concept_id 조립, ROOT 삽입, depth BFS,
                           parent_of/child_of 쌍 생성, A_score 계산,
                           intrinsic_difficulty_level = round(0.4·A + 0.6·B)
  Pass 3 (Embedding, 노드당 1회): text-embedding-004로 768d 벡터 생성

R4와의 계약 (must hold):
  1. nodes[0] = ROOT_DOCUMENT, depth_in_tree=0
  2. ROOT 제외 모든 노드의 parent_concept_id가 채워짐
  3. parent_concept_id가 nodes에 실존 (dangling 없음)
  4. parent_of 엣지는 child_of 쌍을 가짐
  5. 사이클 없음
  6. concept_id 공백·점 없음 (v0.5 네이밍)
  7. 모든 노드 suitable_question_types 최소 1개
  8. intrinsic_difficulty_level 1~5
  9. embeddings_dict.keys() ⊆ {non-root concept_ids}

LOA = 7.
"""
from __future__ import annotations

import functools
import json
import logging
import re
from collections import deque
from pathlib import Path
from typing import Optional

from pydantic import Field, ValidationError

from common.embedding_client import EMBEDDING_DIM, EmbeddingClient
from common.enums import DifficultyLevel, EdgeRelation, Importance, QuestionType
from common.gemini_client import GeminiClient
from common.schemas import (
    ConceptEdge,
    ConceptKnowledgeStructure,
    ConceptNode,
    ConceptSummary,
    StrictBase,
)
from config.defaults import PROJECT_ROOT

logger = logging.getLogger(__name__)

AGENT_NAME = "Topic_Analyzer"
PROMPT_REL_PATH = "config/prompts/topic_analyzer.md"
ROOT_CONCEPT_ID = "ROOT_DOCUMENT"

# v0.5: A_score 가중치
A_WEIGHT_IN_DEG = 0.5
A_WEIGHT_OUT_DEG = 0.3
A_WEIGHT_DEPTH = 0.2

# v0.5: 최종 산출 가중치 (정성 B가 더 무거움)
INTRINSIC_WEIGHT_A = 0.4
INTRINSIC_WEIGHT_B = 0.6

# suitable_question_types 빈 리스트 → level 기반 기본값
DEFAULT_TYPES_BY_LEVEL = {
    DifficultyLevel.L1: [QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
    DifficultyLevel.L2: [QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER],
    DifficultyLevel.L3: [QuestionType.LONG_ANSWER, QuestionType.CASE_ANALYSIS],
    DifficultyLevel.L4: [QuestionType.LONG_ANSWER, QuestionType.CASE_ANALYSIS],
    DifficultyLevel.L5: [QuestionType.CASE_ANALYSIS],
}


# ────────────────────────────────────────────────────────────
# LLM raw output 검증 모델
# ────────────────────────────────────────────────────────────
class RawNode(StrictBase):
    module_id: str
    concept_name: str
    slug: str
    parent_id: str  # ROOT_DOCUMENT 또는 같은 모듈의 concept_id
    B_score: int = Field(ge=1, le=5)
    suitable_question_types: list[str] = Field(default_factory=list)
    importance: str = "medium"
    source_pages: list[str] = Field(default_factory=list)


class RawExtraEdge(StrictBase):
    from_: str = Field(alias="from")
    to: str
    relation: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class TopicAnalyzerRawOutput(StrictBase):
    nodes: list[RawNode] = Field(default_factory=list)
    extra_edges: list[RawExtraEdge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────
# 프롬프트 캐시
# ────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _load_prompt_file(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


# ────────────────────────────────────────────────────────────
# Slug 정제 (v0.5 네이밍 규칙)
# ────────────────────────────────────────────────────────────
def sanitize_slug(raw_slug: str) -> str:
    """LLM이 준 slug를 v0.5 규칙에 강제로 맞춤.

    - 영문 소문자+숫자+`_`만 허용
    - 4단어 이하로 자름
    - 빈 결과면 빈 문자열 반환 (caller가 fallback 처리)

    >>> sanitize_slug("Taylor's 4 Principles")
    'taylor_s_4_principles'
    >>> sanitize_slug("KJ_Method")
    'kj_method'
    >>> sanitize_slug("one_two_three_four_five_six")
    'one_two_three_four'
    >>> sanitize_slug("한글만")
    ''
    """
    s = raw_slug.lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return ""
    parts = s.split("_")
    if len(parts) > 4:
        parts = parts[:4]
    return "_".join(parts)


# ────────────────────────────────────────────────────────────
# 메인 Analyzer 클래스
# ────────────────────────────────────────────────────────────
class TopicAnalyzer:
    """Topic Analyzer Agent.

    Usage:
        analyzer = TopicAnalyzer(GeminiClient(...), EmbeddingClient(...))
        structure, embeddings = analyzer.analyze(summaries, session_id="abc")
    """

    def __init__(
        self,
        client: GeminiClient,
        embedding_client: EmbeddingClient,
    ) -> None:
        self.client = client
        self.embedding_client = embedding_client
        self.system_prompt = _load_prompt_file(
            str(PROJECT_ROOT / PROMPT_REL_PATH)
        )

    # ── 공개 인터페이스 ──────────────────────────────────────
    def analyze(
        self,
        summaries: list[ConceptSummary],
        session_id: str,
    ) -> tuple[ConceptKnowledgeStructure, dict[str, list[float]]]:
        """3-Pass 처리."""
        if not summaries:
            logger.warning("[Topic_Analyzer] 빈 입력 — ROOT만 있는 구조 반환")
            return self._minimal_structure(), {}

        # Pass 1: LLM 호출
        raw = self._call_llm_pass1(summaries)
        if raw is None:
            logger.warning("[Topic_Analyzer] LLM 실패 — Layer 1 fallback")
            return self._layer1_fallback(summaries)

        # Pass 2: Python 결정론 처리
        warnings_acc: list[str] = list(raw.warnings)
        try:
            structure = self._build_structure(raw, summaries, warnings_acc)
        except Exception as exc:
            logger.warning("[Topic_Analyzer] Pass 2 실패: %s — Layer 1 fallback", exc)
            return self._layer1_fallback(summaries)

        # Pass 3: Embedding
        embeddings = self._embed_nodes(structure, summaries)

        return structure, embeddings

    # ── Pass 1: LLM 호출 ────────────────────────────────────
    def _call_llm_pass1(
        self,
        summaries: list[ConceptSummary],
    ) -> Optional[TopicAnalyzerRawOutput]:
        user_prompt = self._build_user_prompt(summaries)
        try:
            raw_json = self.client.generate_json(
                agent_name=AGENT_NAME,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
            )
            return TopicAnalyzerRawOutput.model_validate(raw_json)
        except (ValidationError, RuntimeError, ValueError) as exc:
            logger.warning("[Topic_Analyzer] LLM Pass 1 검증 실패: %s", exc)
            return None

    def _build_user_prompt(self, summaries: list[ConceptSummary]) -> str:
        payload = [
            {
                "module_id": s.module_id,
                "source_filename": s.source_filename,
                "title": s.title,
                "summary_text": s.summary_text,
                "key_phrases": s.key_phrases,
            }
            for s in summaries
        ]
        return json.dumps(payload, ensure_ascii=False)

    # ── Pass 2: Python 결정론 구조 구축 ─────────────────────
    def _build_structure(
        self,
        raw: TopicAnalyzerRawOutput,
        summaries: list[ConceptSummary],
        warnings_acc: list[str],
    ) -> ConceptKnowledgeStructure:
        # 2.1 raw 노드를 concept_id 조립 + 중복 처리
        raw_nodes_with_ids = self._assemble_concept_ids(raw.nodes, warnings_acc)

        if not raw_nodes_with_ids:
            raise ValueError("Pass 1에서 유효한 노드 0개")

        # 2.2 parent_id 검증 (실존하는 concept_id인지)
        valid_ids = {rid for rid, _ in raw_nodes_with_ids} | {ROOT_CONCEPT_ID}
        for rid, rn in raw_nodes_with_ids:
            if rn.parent_id not in valid_ids:
                warnings_acc.append(
                    f"노드 '{rid}'의 parent_id '{rn.parent_id}' 미존재 → ROOT_DOCUMENT로 강제"
                )
                rn.parent_id = ROOT_CONCEPT_ID

        # 2.3 사이클 검증
        self._check_no_cycles(raw_nodes_with_ids, warnings_acc)

        # 2.4 depth BFS 계산
        depth_map = self._compute_depths(raw_nodes_with_ids)

        # 2.5 노드별 in/out degree (parent_of/child_of 쌍 + extra_edges 모두 고려)
        in_deg, out_deg = self._compute_degrees(raw_nodes_with_ids, raw.extra_edges)

        # 2.6 A_score · 최종 intrinsic level 산출
        a_scores = self._compute_a_scores(raw_nodes_with_ids, in_deg, out_deg, depth_map)

        # 2.7 ConceptNode 객체 생성
        nodes: list[ConceptNode] = [self._make_root_node()]
        for rid, rn in raw_nodes_with_ids:
            b = float(rn.B_score)
            a = a_scores.get(rid, 0.0)
            final_val = INTRINSIC_WEIGHT_A * a + INTRINSIC_WEIGHT_B * b
            level_int = max(1, min(5, round(final_val)))
            level = DifficultyLevel(level_int)

            types = self._resolve_question_types(rn.suitable_question_types, level)
            importance = self._resolve_importance(rn.importance)

            nodes.append(ConceptNode(
                concept_id=rid,
                concept_name=rn.concept_name,
                depth_in_tree=depth_map.get(rid, 1),
                parent_concept_id=rn.parent_id,
                importance=importance,
                intrinsic_difficulty_level=level,
                suitable_question_types=types,
                source_pages=rn.source_pages,
                embedding_vector=None,
            ))

        # 2.8 엣지 생성 (parent_of/child_of 쌍 + extra_edges)
        edges = self._build_edges(raw_nodes_with_ids, raw.extra_edges, valid_ids, warnings_acc)

        return ConceptKnowledgeStructure(nodes=nodes, edges=edges)

    def _assemble_concept_ids(
        self,
        raw_nodes: list[RawNode],
        warnings_acc: list[str],
    ) -> list[tuple[str, RawNode]]:
        """concept_id 조립 + 중복 처리. 정제 실패한 노드는 버림."""
        results: list[tuple[str, RawNode]] = []
        used_ids: set[str] = set()

        for idx, rn in enumerate(raw_nodes):
            slug = sanitize_slug(rn.slug)
            if not slug:
                fallback = f"concept_{idx}"
                warnings_acc.append(
                    f"노드 '{rn.concept_name}' slug 정제 실패 → '{fallback}' 사용"
                )
                slug = fallback

            base_id = f"{rn.module_id}_{slug}"
            cid = base_id
            v = 2
            while cid in used_ids:
                cid = f"{base_id}_v{v}"
                v += 1
            used_ids.add(cid)
            results.append((cid, rn))

        return results

    def _check_no_cycles(
        self,
        raw_with_ids: list[tuple[str, RawNode]],
        warnings_acc: list[str],
    ) -> None:
        """부모 체인을 따라가 사이클 검출. 발견 시 ROOT로 끊음."""
        parent_of = {cid: rn.parent_id for cid, rn in raw_with_ids}
        for start_cid in list(parent_of.keys()):
            visited = set()
            current = start_cid
            while current != ROOT_CONCEPT_ID and current in parent_of:
                if current in visited:
                    warnings_acc.append(
                        f"순환 참조 발견 (시작: '{start_cid}') → '{current}' 부모를 ROOT로 강제"
                    )
                    parent_of[current] = ROOT_CONCEPT_ID
                    # raw_with_ids에도 반영
                    for cid, rn in raw_with_ids:
                        if cid == current:
                            rn.parent_id = ROOT_CONCEPT_ID
                    break
                visited.add(current)
                current = parent_of[current]

    def _compute_depths(
        self,
        raw_with_ids: list[tuple[str, RawNode]],
    ) -> dict[str, int]:
        """ROOT부터 BFS로 depth 계산."""
        depths = {ROOT_CONCEPT_ID: 0}
        children_of: dict[str, list[str]] = {ROOT_CONCEPT_ID: []}
        for cid, rn in raw_with_ids:
            children_of.setdefault(rn.parent_id, []).append(cid)
            children_of.setdefault(cid, [])

        queue = deque([ROOT_CONCEPT_ID])
        while queue:
            current = queue.popleft()
            for child in children_of.get(current, []):
                if child not in depths:
                    depths[child] = depths[current] + 1
                    queue.append(child)

        # BFS에서 안 닿은 노드(disconnected) → depth=1로 강제 (이미 cycle 검사 후이므로 드뭄)
        for cid, _ in raw_with_ids:
            depths.setdefault(cid, 1)
        return depths

    def _compute_degrees(
        self,
        raw_with_ids: list[tuple[str, RawNode]],
        extra_edges: list[RawExtraEdge],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """parent_of/child_of 쌍 + extra_edges 기준 in/out_degree."""
        in_deg: dict[str, int] = {ROOT_CONCEPT_ID: 0}
        out_deg: dict[str, int] = {ROOT_CONCEPT_ID: 0}
        for cid, _ in raw_with_ids:
            in_deg[cid] = 0
            out_deg[cid] = 0

        # parent → child (parent_of) + child → parent (child_of)
        for cid, rn in raw_with_ids:
            # parent_of: parent → child
            in_deg[cid] = in_deg.get(cid, 0) + 1
            out_deg[rn.parent_id] = out_deg.get(rn.parent_id, 0) + 1
            # child_of: child → parent
            in_deg[rn.parent_id] = in_deg.get(rn.parent_id, 0) + 1
            out_deg[cid] = out_deg.get(cid, 0) + 1

        # extra_edges (related/prerequisite)
        valid_ids = {cid for cid, _ in raw_with_ids} | {ROOT_CONCEPT_ID}
        for ee in extra_edges:
            if ee.from_ not in valid_ids or ee.to not in valid_ids:
                continue
            out_deg[ee.from_] = out_deg.get(ee.from_, 0) + 1
            in_deg[ee.to] = in_deg.get(ee.to, 0) + 1
            # prerequisite은 depends_on 쌍을 만들므로 역방향도 카운트
            if ee.relation == "prerequisite":
                out_deg[ee.to] = out_deg.get(ee.to, 0) + 1
                in_deg[ee.from_] = in_deg.get(ee.from_, 0) + 1

        return in_deg, out_deg

    def _compute_a_scores(
        self,
        raw_with_ids: list[tuple[str, RawNode]],
        in_deg: dict[str, int],
        out_deg: dict[str, int],
        depth_map: dict[str, int],
    ) -> dict[str, float]:
        """v0.5 공식: 0.5·norm(in)·5 + 0.3·norm(out)·5 + 0.2·norm(depth)·5"""
        max_in = max((in_deg.get(c, 0) for c, _ in raw_with_ids), default=0) or 1
        max_out = max((out_deg.get(c, 0) for c, _ in raw_with_ids), default=0) or 1
        max_depth = max((depth_map.get(c, 0) for c, _ in raw_with_ids), default=0) or 1

        scores: dict[str, float] = {}
        for cid, _ in raw_with_ids:
            norm_in = in_deg.get(cid, 0) / max_in
            norm_out = out_deg.get(cid, 0) / max_out
            norm_depth = depth_map.get(cid, 0) / max_depth
            a = (
                A_WEIGHT_IN_DEG * norm_in * 5.0
                + A_WEIGHT_OUT_DEG * norm_out * 5.0
                + A_WEIGHT_DEPTH * norm_depth * 5.0
            )
            scores[cid] = a
        return scores

    def _resolve_question_types(
        self,
        raw_types: list[str],
        level: DifficultyLevel,
    ) -> list[QuestionType]:
        """LLM이 준 유형 string을 enum으로 변환. 빈 리스트면 level 기반 기본값."""
        result: list[QuestionType] = []
        for t in raw_types:
            try:
                result.append(QuestionType(t))
            except ValueError:
                continue
        if not result:
            return list(DEFAULT_TYPES_BY_LEVEL[level])
        return result

    def _resolve_importance(self, raw: str) -> Importance:
        try:
            return Importance(raw.lower())
        except ValueError:
            return Importance.MEDIUM

    def _build_edges(
        self,
        raw_with_ids: list[tuple[str, RawNode]],
        extra_edges: list[RawExtraEdge],
        valid_ids: set[str],
        warnings_acc: list[str],
    ) -> list[ConceptEdge]:
        edges: list[ConceptEdge] = []

        # parent_of + child_of 쌍 (Rule A)
        for cid, rn in raw_with_ids:
            edges.append(ConceptEdge(
                from_concept_id=rn.parent_id,
                to_concept_id=cid,
                relation=EdgeRelation.PARENT_OF,
                weight=1.0,
            ))
            edges.append(ConceptEdge(
                from_concept_id=cid,
                to_concept_id=rn.parent_id,
                relation=EdgeRelation.CHILD_OF,
                weight=1.0,
            ))

        # extra_edges 처리
        for ee in extra_edges:
            if ee.from_ not in valid_ids or ee.to not in valid_ids:
                warnings_acc.append(
                    f"extra_edge '{ee.from_}→{ee.to}': 노드 미존재 → 무시"
                )
                continue
            try:
                relation = EdgeRelation(ee.relation)
            except ValueError:
                warnings_acc.append(f"extra_edge relation '{ee.relation}' 미지원 → 무시")
                continue

            if relation == EdgeRelation.RELATED:
                # Rule C: 단방향 단일 엣지
                edges.append(ConceptEdge(
                    from_concept_id=ee.from_,
                    to_concept_id=ee.to,
                    relation=EdgeRelation.RELATED,
                    weight=ee.weight,
                ))
            elif relation == EdgeRelation.PREREQUISITE:
                # Rule B: prerequisite + depends_on 쌍
                edges.append(ConceptEdge(
                    from_concept_id=ee.from_,
                    to_concept_id=ee.to,
                    relation=EdgeRelation.PREREQUISITE,
                    weight=ee.weight,
                ))
                edges.append(ConceptEdge(
                    from_concept_id=ee.to,
                    to_concept_id=ee.from_,
                    relation=EdgeRelation.DEPENDS_ON,
                    weight=ee.weight,
                ))

        return edges

    # ── Pass 3: Embedding ───────────────────────────────────
    def _embed_nodes(
        self,
        structure: ConceptKnowledgeStructure,
        summaries: list[ConceptSummary],
    ) -> dict[str, list[float]]:
        # 모듈별 요약 텍스트 dict
        summary_text_by_module = {s.module_id: s.summary_text for s in summaries}

        embeddings: dict[str, list[float]] = {}
        for node in structure.nodes:
            if node.concept_id == ROOT_CONCEPT_ID:
                continue
            module_id = node.concept_id.split("_")[0]
            # M1_4 형태 module_id 복원: split 후 앞 1~3개 토큰
            mod_prefix = self._infer_module_id_from_concept_id(node.concept_id)
            module_summary = summary_text_by_module.get(mod_prefix, "")
            text = f"{node.concept_name}\n{module_summary[:500]}"

            vec = self.embedding_client.embed(
                agent_name=AGENT_NAME,
                text=text,
            )
            embeddings[node.concept_id] = vec

        return embeddings

    @staticmethod
    def _infer_module_id_from_concept_id(concept_id: str) -> str:
        """concept_id에서 module_id 부분 추출.

        - "M1_4_taylor_principles" → "M1_4"
        - "M2_1_2_kj_method" → "M2_1_2"
        - "manufacturing_overview" → "manufacturing_overview"
        """
        parts = concept_id.split("_")
        if not parts:
            return concept_id
        # M으로 시작 → 뒤에 숫자가 이어지는 만큼 module_id에 포함
        if parts[0].startswith("M") and parts[0][1:].isdigit():
            head = [parts[0]]
            i = 1
            while i < len(parts) and parts[i].isdigit():
                head.append(parts[i])
                i += 1
            return "_".join(head)
        return concept_id

    # ── Helpers ─────────────────────────────────────────────
    def _make_root_node(self) -> ConceptNode:
        return ConceptNode(
            concept_id=ROOT_CONCEPT_ID,
            concept_name="(root)",
            depth_in_tree=0,
            parent_concept_id=None,
            importance=Importance.LOW,
            intrinsic_difficulty_level=DifficultyLevel.L1,
            suitable_question_types=[],
            source_pages=[],
            embedding_vector=None,
        )

    def _minimal_structure(self) -> ConceptKnowledgeStructure:
        """빈 입력 → ROOT만 있는 구조."""
        return ConceptKnowledgeStructure(nodes=[self._make_root_node()], edges=[])

    # ── Layer 1 Fallback: 모듈당 root 노드 1개씩 ─────────────
    def _layer1_fallback(
        self,
        summaries: list[ConceptSummary],
    ) -> tuple[ConceptKnowledgeStructure, dict[str, list[float]]]:
        """LLM 실패 시: 각 ConceptSummary마다 overview 노드 1개씩."""
        nodes: list[ConceptNode] = [self._make_root_node()]
        edges: list[ConceptEdge] = []
        embeddings: dict[str, list[float]] = {}

        for s in summaries:
            cid = f"{s.module_id}_overview"
            nodes.append(ConceptNode(
                concept_id=cid,
                concept_name=s.title or s.module_id,
                depth_in_tree=1,
                parent_concept_id=ROOT_CONCEPT_ID,
                importance=Importance.MEDIUM,
                intrinsic_difficulty_level=DifficultyLevel.L3,
                suitable_question_types=[
                    QuestionType.LONG_ANSWER,
                    QuestionType.CASE_ANALYSIS,
                ],
                source_pages=[f"{s.source_filename}"],
                embedding_vector=None,
            ))
            edges.append(ConceptEdge(
                from_concept_id=ROOT_CONCEPT_ID,
                to_concept_id=cid,
                relation=EdgeRelation.PARENT_OF,
                weight=1.0,
            ))
            edges.append(ConceptEdge(
                from_concept_id=cid,
                to_concept_id=ROOT_CONCEPT_ID,
                relation=EdgeRelation.CHILD_OF,
                weight=1.0,
            ))
            vec = self.embedding_client.embed(
                agent_name=AGENT_NAME,
                text=f"{s.title}\n{s.summary_text[:500]}",
            )
            embeddings[cid] = vec

        return (
            ConceptKnowledgeStructure(nodes=nodes, edges=edges),
            embeddings,
        )


# ────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import doctest
    failed, total = doctest.testmod(verbose=False)
    print(f"✓ doctests: {total - failed}/{total} passed")

    # sanitize_slug 추가 case
    cases = [
        ("Taylor's 4 Principles", "taylor_s_4_principles"),
        ("KJ_Method", "kj_method"),
        ("a__b___c", "a_b_c"),
        ("한글만", ""),
        ("PIG IRON CASE", "pig_iron_case"),
    ]
    for raw, expected in cases:
        actual = sanitize_slug(raw)
        mark = "✓" if actual == expected else "✗"
        print(f"  {mark} sanitize({raw!r}) → {actual!r} (expected {expected!r})")
