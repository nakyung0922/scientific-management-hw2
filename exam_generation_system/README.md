# Agentic Exam Generation System

2026-1 Scientific Management — HW2 Team 7

LLM 기반 multi-agent 시스템으로, 강의자료와 교수 요구사항을 입력받아 시험 문항과 모범답안을 자동 생성합니다.

## 시스템 구조

총 10개의 agent + Human Reviewer로 구성된 agentic work system:

- **PG1 (R3)**: Material Collector, Req Parser, Topic Analyzer, Topic Prioritizer
- **PG2 (R4)**: Exam Planner, Q&A Generator, Factfulness Tester, Difficulty Tester, Rubric Machine, Paper Formattor
- **PG3 (R5)**: Supervisor (orchestration)

자세한 설계는 `agentic_system_schema_v0.5.json` 참고.

## 설치 및 실행

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 GCP 프로젝트 ID와 서비스 계정 키 경로 입력

# 4. GCP 인증 (둘 중 하나 선택)
gcloud auth application-default login
# 또는 GOOGLE_APPLICATION_CREDENTIALS에 service account key 파일 절대 경로 설정
```

## 디렉토리 구조

```
exam_generation_system/
├── config/         # 설정값 + system prompts (LLM 지시사항)
├── common/         # 공통 모듈 (Pydantic 스키마, Gemini wrapper, 난이도 계산)
├── agents/         # 각 agent 구현 (PG1·PG2·PG3)
├── tests/          # 단위 테스트
└── outputs/        # 생성된 시험지·답안지 출력 (.docx)
```

## 사용 예시

### API 연동 테스트

```bash
cd exam_generation_system
python live_api_test.py
```

ExamPlanner, FactfulnessTester, DifficultyTester, RubricMachine의 Vertex AI Gemini 연동을 순차적으로 검증합니다.
실행 전 `.env`에 `GCP_PROJECT_ID`와 `GOOGLE_APPLICATION_CREDENTIALS`를 반드시 설정해야 합니다.

### 전체 파이프라인 실행 (HITL 강화 적용)

```python
from common.gemini_client import GeminiClient
from agents.supervisor import Supervisor
from r6_hitl_enhancer import patch_supervisor

client = GeminiClient(mock=False)
supervisor = Supervisor(client)
patch_supervisor(supervisor)  # MAX_RETRY 초과 슬롯 강조 표시 활성화

result = supervisor.run(
    pdf_paths=["M1_1.pdf", "M2_1.pdf", "M3_1.pdf"],
    requirements="시험 시간: 75분, 총점: 100점, 문항: 단답 2 / 서술 5 / 사례 3",
    hitl=True,
)
print(f"상태: {result.status}, 소요: {result.elapsed_sec}초")
```

### 자동화율 측정

```python
from agents.supervisor import measure_automation_rate
metrics = measure_automation_rate(result)
print(f"자동화율(step 기준): {metrics['step_auto_rate']}%")  # 목표: 80% 이상
print(f"80% 기준 충족 여부: {metrics['meets_80_percent']}")
```
