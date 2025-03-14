# Gemma 에이전트 챗봇 시스템

이 프로젝트는 Ollama의 Gemma 모델을 사용하여 함수 호출(Function Calling) 기능을 구현한 에이전트 챗봇 시스템입니다.

## 프로젝트 구조

├── ollama_agent_app.py # 에이전트 챗봇 애플리케이션
├── requirements.txt # 필요한 패키지 목록
└── README_AGENT_OLLAMA_APP.md # 문서

## 주요 기능

### 1. 기본 함수 기능
- `get_current_weather`: 특정 지역의 날씨 정보 조회
- `get_current_time`: 현재 시간 확인
- `calculate`: 간단한 수식 계산

### 2. 자연어 처리 기능
- 사용자의 자연어 입력을 함수 호출로 변환
- 함수 실행 결과를 자연스러운 한국어 응답으로 변환
- 일반 대화와 함수 호출 의도 구분

### 3. 에이전트 시스템
- 함수 호출 의도 파악
- 적절한 함수 실행
- 결과 포맷팅 및 응답 생성

## 설치 방법

### 1. 사전 요구사항
- Python 3.7 이상
- Docker 및 Docker Compose
- Ollama 서비스 실행 중

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. Ollama 서비스 확인
```bash
curl http://localhost:11434/api/tags
```

## 사용 방법

### 1. 애플리케이션 실행
```bash
python ollama_agent_app.py
```

### 2. 사용 가능한 명령어
- 날씨 확인: "서울의 날씨 알려줘"
- 시간 확인: "지금 시간이 몇 시야?"
- 계산: "23 + 45 계산해줘"
- 종료: 'q' 또는 'quit' 입력

## 코드 구조 설명

### 1. GemmaAgent 클래스
```python
class GemmaAgent:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "gemma:3b"
        self.available_functions = {
            "get_current_weather": self.get_current_weather,
            "get_current_time": self.get_current_time,
            "calculate": self.calculate
        }
```

### 2. 함수 호출 처리
```python
def extract_function_call(self, text: str) -> Optional[Dict]:
    # 텍스트에서 함수 호출 의도를 파악하여 JSON 형식으로 반환
```

### 3. 함수 실행
```python
def execute_function(self, function_info: Dict) -> str:
    # 파악된 함수를 실행하고 결과 반환
```

## 사용 예시

```

</rewritten_file>

## 성능 최적화

1. 함수 호출 인식 개선
   - 프롬프트 엔지니어링을 통한 정확도 향상
   - 컨텍스트 관리를 통한 일관성 유지

2. 응답 생성 최적화
   - 스트리밍 응답 지원
   - 응답 포맷팅 개선

3. 오류 처리
   - 함수 호출 실패 처리
   - 잘못된 입력 처리

## 제한사항 및 주의사항

1. 함수 인식 정확도
   - 모델의 함수 인식 능력에 따라 정확도가 달라질 수 있음
   - 복잡한 함수 호출은 인식이 어려울 수 있음

2. 보안
   - eval() 함수 사용에 따른 보안 위험
   - 실제 환경에서는 안전한 구현 필요

3. 성능
   - 함수 호출 인식에 추가 API 호출 필요
   - 응답 시간이 다소 길어질 수 있음

## 향후 개선 사항

1. 기능 확장
   - 더 많은 유틸리티 함수 추가
   - 복잡한 함수 호출 지원
   - 대화 컨텍스트 관리 개선

2. 성능 개선
   - 함수 호출 인식 정확도 향상
   - 응답 시간 최적화
   - 캐싱 구현

3. 보안 강화
   - 안전한 함수 실행 메커니즘
   - 입력 검증 강화
   - 권한 관리 추가

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.