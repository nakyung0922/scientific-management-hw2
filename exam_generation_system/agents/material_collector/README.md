# Material Collector

PG1 (R3) — **Information Extractor** 그룹의 첫 번째 에이전트.
강의자료 PDF → 개념 단위 (Topic Analyzer 가 ConceptNode 로 변환할 재료).

## v0.5 명세 매핑

| 항목 | 값 |
| --- | --- |
| agent_id | `Material_Collector` |
| input | `exam_generation_system/agents/material_collector/materials/` — 강의자료 PDF |
| output | `MaterialCollectorOutput` (Pydantic, payload of MessageEnvelope) |
| next agent | `Topic_Analyzer` |
| routing_status | `flow` |
| concept_id rule | `{module_id}_{snake_case_slug}`, 영문/숫자/_ 만 (v0.5) |
| LLM | Gemini Vertex AI via `common.gemini_client.GeminiClient` |

## 디렉토리

```
agents/material_collector/
├── __init__.py            # public API
├── agent.py               # MaterialCollector 클래스 (조립)
├── schemas.py             # MaterialCollectorOutput, ConceptUnit
├── pdf_reader.py          # PyMuPDF + Tesseract OCR fallback
├── chunker.py             # 페이지 → 의미 단위 청크
├── prompts.py             # Gemini system/user prompt
├── summarizer.py          # GeminiClient 호출 래퍼
├── concept_id.py          # v0.5 네이밍 규칙 구현
└── run.py                 # CLI 진입점
```

테스트: `tests/test_material_collector.py` (11개 케이스, PG2 패턴)

## 설계 원칙: 시험 범위에 가정하지 않음

Material Collector 는 입력으로 들어오는 PDF 가 어떤 챕터인지에 대해 가정하지 않는다.
어떤 PDF 가 들어오든 일관된 규칙으로 라벨링하고 처리한다. 시험 범위와 들어온
자료의 매칭은 상위 에이전트 (Exam_Planner) 의 책임이다.

### 파일명 → module_id 라벨링 규칙 (`label_module_from_filename`)

| 입력 파일명 | 라벨 (module_id) | 설명 |
| --- | --- | --- |
| `M1.4.pdf` | `M1_4` | 점 → 언더스코어 |
| `M2.1.2_1.pdf` | `M2_1_2_1` | 분할된 파일도 그대로 |
| `M5_1_Automation.pdf` | `M5_1_automation` | 코드 + 영문 합성 |
| `IntroductionOverviewManufacturing.pdf` | `introduction_overview_manufacturing` | CamelCase 분해 |
| `KJ Method.pdf` | `kj_method` | 공백/대소문자 정규화 |
| `자동화_개론.pdf` | `material_<hash>` + `auto_labeled` warning | ASCII 없으면 안정적 해시 |
| 같은 라벨 두 번 | `M1_4`, `M1_4_2`, `M1_4_3` | 충돌 회피 |

라벨링 결과는 payload 의 `file_to_module_id` 매핑에 저장돼 Topic_Analyzer 가
어떤 PDF 가 어떤 module 인지 추적할 수 있다.

## 설치 (PG2 requirements.txt 위에 추가)

```bash
# PG2 베이스 의존성 (이미 설치돼 있을 것)
pip install -r requirements.txt

# Material Collector 추가 의존성
pip install pymupdf pytesseract pillow

# OCR (이미지 기반 PDF 대응 — 강의자료가 PPT export 인 경우 필수)
apt-get install -y tesseract-ocr tesseract-ocr-kor
```

requirements.txt 에 PG1 영역으로 추가해야 할 항목:
```
pymupdf>=1.24.0       # PG1 영역 — pypdf 보다 정확하고 OCR 친화적
pytesseract>=0.3.10   # PG1 영역 — 이미지 기반 PDF 대응
pillow>=10.0.0
```

> 기존 requirements 의 `pypdf>=4.0.0` 항목은 PyMuPDF 로 대체하기를 제안.
> 이유: 강의자료가 PPT export PDF 인 경우 pypdf 로 텍스트 추출이 0자.

## 사용법

### A) Python 에서 import

```python
from common.gemini_client import GeminiClient
from agents.material_collector import (
    MaterialCollector,
    MaterialCollectorConfig,
)

client = GeminiClient(mock=False)  # 또는 mock=True (단위 테스트)
agent = MaterialCollector(
    client=client,
    config=MaterialCollectorConfig(
        pages_per_chunk=2,
        enable_ocr=True,
        ocr_lang="kor+eng",
    ),
)

env = agent.run(
    pdf_paths=["./M1_4_taylorism.pdf", "./M2_1_5_kj_method.pdf"],
    session_id="session_001",
)

# env 는 MessageEnvelope. 그대로 Topic_Analyzer 로 전달.
print(env.target_agent)        # "Topic_Analyzer"
print(env.payload["total_concepts"])
```

### B) CLI

```bash
# Vertex AI 실 호출
export GCP_PROJECT_ID=...
gcloud auth application-default login
python -m agents.material_collector.run \
    ./M1_4_taylorism.pdf \
    --session session_001 \
    --out output.json

# Mock 모드 (GCP 없이)
python -m agents.material_collector.run ./M1_4.pdf --mock --out test.json

# LLM 완전 우회 (휴리스틱, 빠른 디버깅)
python -m agents.material_collector.run ./M1_4.pdf --no-llm --out test.json
```

## 출력 구조

`env.payload` = `MaterialCollectorOutput` 의 dump:

```jsonc
{
  "session_id": "session_001",
  "generated_by": "Material_Collector",
  "course_name": "Scientific Management",
  "source_files": ["M1.4.pdf", "IntroductionOverviewManufacturing.pdf"],
  "total_pages": 103,
  "total_concepts": 50,
  "file_to_module_id": {
    "M1.4.pdf": "M1_4",
    "IntroductionOverviewManufacturing.pdf": "introduction_overview_manufacturing"
  },
  "extraction_methods": {"text": 103},
  "ocr_languages": [],
  "warnings": [],
  "concepts": [
    {
      "concept_id": "M1_4_taylor_principles",
      "concept_name": "taylor principles",
      "summary": "테일러의 4가지 원칙은 ...",
      "keywords": ["scientific management", "테일러", "4 원칙"],
      "source_file": "M1.4.pdf",
      "source_pages": ["M1_4 p.3", "M1_4 p.4"],
      "raw_text": "<원문 전체 — RAG 근거 인용용>",
      "section_title": "Taylor's 4 Principles",
      "extraction_confidence": 1.0
    }
  ]
}
```

## Topic_Analyzer 와의 인터페이스 계약

`ConceptUnit` 의 모든 필드는 `common.schemas.ConceptNode` 로 매끄럽게 변환된다:

| ConceptUnit | → | ConceptNode |
| --- | --- | --- |
| `concept_id` | → | `concept_id` (이미 v0.5 규칙 통과) |
| `concept_name` | → | `concept_name` |
| `source_pages` | → | `source_pages` (동일 형식) |
| `summary` + `keywords` | → | 임베딩 입력 |
| `raw_text` | → | RAG 근거 인용 (Hallucination 검증) |

`depth_in_tree`, `parent_concept_id`, `importance`, `intrinsic_difficulty_level`,
`suitable_question_types` 는 Topic_Analyzer 가 그래프 구축하면서 채운다.

## 설계 결정 기록

- **시험 범위에 가정하지 않음**: 이전 버전은 `target_chapters` 에 들어있지 않은
  파일에 대해 warning 을 띄웠으나, Material Collector 의 책임은 "들어온 자료를
  라벨링·구조화하는 것" 이고 시험 범위 매칭은 상위 에이전트의 책임이므로 폐기.
  `module_id_in_target_chapters()` 함수와 `TARGET_CHAPTERS` import 도 함께 제거.
- **CamelCase 파일명 자동 분해**: `IntroductionOverviewManufacturing.pdf` 같은
  파일을 `introduction_overview_manufacturing` 으로 변환. 다른 module 코드 패턴
  (`M1.4`) 과도 충돌하지 않음.
- **한국어 파일명 fallback**: 안정적 해시 (sha1 prefix) 로 `material_<hash>`
  라벨링. `auto_labeled` 정보성 warning 으로 호출자가 알 수 있도록 함.
- **PyMuPDF + OCR**: 강의자료가 PPT export PDF 인 경우 일반 텍스트 추출이
  0자임을 실측 확인. Tesseract `kor+eng` 로 36 페이지 OCR 시 약 2분.
- **concept_name 은 영어 강제**: v0.5 명세가 영문/숫자/_ 만 허용하므로,
  LLM 프롬프트가 "concept_name MUST be ENGLISH" 를 강제. 한글 summary/keywords 는 유지.
- **failure isolation**: 한 파일이 실패해도 다른 파일 처리 계속. warnings 에 기록.
- **extraction_confidence**: OCR 페이지(0.7) / LLM fallback(0.5) 자동 감소 →
  다음 에이전트가 신뢰도 가중 가능.
