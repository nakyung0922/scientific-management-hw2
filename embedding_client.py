"""
Vertex AI text-embedding-004 wrapper.

GeminiClient와 동일한 패턴이지만 별도 클래스인 이유:
- 생성 모델과 임베딩 모델은 API 호출 방식·요금·rate limit이 다름
- Topic_Analyzer Pass 3에서만 사용 (제한된 범위)
- Mock 모드의 의미가 다름 (텍스트 생성 vs 768d 벡터 반환)
- usage_tracker는 GeminiClient의 것을 공유 (단일 회계)

설계 원칙:
- 노드당 1회 호출 (배치 미사용 — 단순함 우선)
- 768차원 고정 (text-embedding-004 표준)
- Mock 모드는 텍스트 해시 기반 결정론 벡터 (재현 가능 테스트)
- 실패 시 raise하지 않고 zero vector 반환 (Topic_Analyzer가 degraded mode로 동작)
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Callable, Optional

from common.gemini_client import UsageRecord, _tracker
from config.defaults import DEBUG_MODE, LLM_CONFIG

logger = logging.getLogger(__name__)
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG)


# ────────────────────────────────────────────────────────────
# 상수
# ────────────────────────────────────────────────────────────
EMBEDDING_DIM = 768  # text-embedding-004 표준
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"


# ────────────────────────────────────────────────────────────
# Client
# ────────────────────────────────────────────────────────────
class EmbeddingClient:
    """Vertex AI text-embedding-004 wrapper.

    Usage:
        client = EmbeddingClient(mock=True)
        vec = client.embed(agent_name="Topic_Analyzer", text="Taylor's principles")
        # len(vec) == 768

    실 호출은 google-genai SDK 사용 (Vertex AI 모드).
    mock=True로 생성하면 텍스트 해시 기반 결정론 벡터 반환.
    """

    def __init__(
        self,
        *,
        mock: bool = False,
        mock_responder: Optional[Callable[[str], list[float]]] = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Args:
            mock: True면 GCP 호출 없이 결정론 벡터 반환.
            mock_responder: mock 모드 시 (text) → 768d list[float] 함수.
                            None이면 텍스트 해시 기반 기본 responder.
            model: 임베딩 모델명. 기본 text-embedding-004.
        """
        self.mock = mock
        self.mock_responder = mock_responder or _hash_based_mock
        self.model = model

        if not mock:
            self._client = self._init_real_client()

    def _init_real_client(self):
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "google-genai가 설치되지 않았습니다. "
                "pip install google-genai 를 실행하세요."
            ) from e

        project_id = LLM_CONFIG["project_id"]
        region = LLM_CONFIG["region"]

        if not project_id:
            raise RuntimeError(
                "GCP_PROJECT_ID가 설정되지 않았습니다. "
                ".env 파일을 확인하거나 mock=True로 생성하세요."
            )

        return genai.Client(
            vertexai=True,
            project=project_id,
            location=region,
        )

    def embed(
        self,
        *,
        agent_name: str,
        text: str,
        max_retries: int = 3,
    ) -> list[float]:
        """단일 텍스트를 768d 벡터로 변환.

        실패 시 zero vector 반환 (raise 안 함). Topic_Analyzer가
        degraded mode로 동작할 수 있도록.
        """
        start = time.time()

        if self.mock:
            vec = self.mock_responder(text)
            duration = time.time() - start
            _tracker.add(UsageRecord(
                agent_name=agent_name,
                model=f"mock:{self.model}",
                duration_seconds=duration,
            ))
            return vec

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                vec = self._call_real_api(text)
                _tracker.add(UsageRecord(
                    agent_name=agent_name,
                    model=self.model,
                    duration_seconds=time.time() - start,
                ))
                return vec
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{agent_name}] Embedding error (attempt {attempt}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(1.0 * attempt)

        _tracker.add(UsageRecord(
            agent_name=agent_name,
            model=self.model,
            duration_seconds=time.time() - start,
            success=False,
        ))
        logger.warning(
            f"[{agent_name}] Embedding {max_retries}회 실패: {last_error}. "
            f"Zero vector 반환."
        )
        return [0.0] * EMBEDDING_DIM

    def _call_real_api(self, text: str) -> list[float]:
        """실제 google-genai 임베딩 API 호출."""
        response = self._client.models.embed_content(
            model=self.model,
            contents=text,
        )
        # google-genai 응답 구조: response.embeddings[0].values
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise RuntimeError("Empty embeddings response")
        values = embeddings[0].values
        if len(values) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Expected {EMBEDDING_DIM}d vector, got {len(values)}d"
            )
        return list(values)


# ────────────────────────────────────────────────────────────
# Default mock responder — 해시 기반 결정론
# ────────────────────────────────────────────────────────────
def _hash_based_mock(text: str) -> list[float]:
    """텍스트의 SHA256을 시드로 한 결정론 768d 벡터.

    같은 텍스트 → 같은 벡터. 다른 텍스트 → 다른 벡터.
    [-1, 1] 범위. 단위 길이는 보장하지 않음.

    >>> v1 = _hash_based_mock("hello")
    >>> v2 = _hash_based_mock("hello")
    >>> v1 == v2
    True
    >>> v3 = _hash_based_mock("world")
    >>> v1 == v3
    False
    >>> len(v1) == 768
    True
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # 32 bytes를 768 floats로 확장: 반복 hash chain
    vec: list[float] = []
    current = digest
    while len(vec) < EMBEDDING_DIM:
        current = hashlib.sha256(current).digest()
        # 32 bytes → 32 floats in [-1, 1]
        for b in current:
            vec.append((b / 127.5) - 1.0)
            if len(vec) >= EMBEDDING_DIM:
                break
    return vec


# ────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import doctest
    failed, total = doctest.testmod(verbose=False)
    print(f"✓ doctests: {total - failed}/{total} passed")

    client = EmbeddingClient(mock=True)
    v1 = client.embed(agent_name="Test", text="Taylor's scientific management")
    v2 = client.embed(agent_name="Test", text="KJ Method clustering")
    v3 = client.embed(agent_name="Test", text="Taylor's scientific management")

    assert len(v1) == EMBEDDING_DIM
    assert v1 == v3, "결정론 깨짐"
    assert v1 != v2, "다른 텍스트가 같은 벡터"
    print(f"✓ Embedding mock: {EMBEDDING_DIM}d, deterministic, distinct")
