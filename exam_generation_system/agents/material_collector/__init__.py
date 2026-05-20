"""
Material Collector agent — PG1 (R3) Information Extractor.

Public API:
    MaterialCollector       : 메인 에이전트 클래스
    MaterialCollectorConfig : 실행 설정
    MaterialCollectorOutput : 출력 payload Pydantic 모델
    ConceptUnit             : 개별 개념 단위 모델
    AGENT_NAME              : "Material_Collector"
    NEXT_AGENT              : "Topic_Analyzer"
"""
from agents.material_collector.agent import (
    AGENT_NAME,
    NEXT_AGENT,
    MaterialCollector,
    MaterialCollectorConfig,
)
from agents.material_collector.schemas import (
    ConceptUnit,
    MaterialCollectorOutput,
)

__all__ = [
    "MaterialCollector",
    "MaterialCollectorConfig",
    "MaterialCollectorOutput",
    "ConceptUnit",
    "AGENT_NAME",
    "NEXT_AGENT",
]
