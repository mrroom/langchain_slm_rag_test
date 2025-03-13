# Ollama Gemma 챗봇 애플리케이션

이 프로젝트는 Langchain과 Ollama를 사용하여 Gemma 3B 모델 기반의 한국어 챗봇을 구현한 애플리케이션입니다.

## 프로젝트 구조 

```
.
├── ollama_app.py        # 메인 애플리케이션 코드
├── requirements.txt     # 필요한 패키지 목록
└── README_OLLAMA_APP.md # 문서
```

## 주요 기능

### 1. GemmaChatBot 클래스
- 모델 초기화 및 설정
- 대화 기록 관리
- 사용자 입력 처리 및 응답 생성
- 스트리밍 방식의 실시간 응답 출력

### 2. 대화 관리 기능
- 대화 기록 저장 및 불러오기
- 대화 기록 초기화 (`clear` 명령어)
- 시스템 프롬프트를 통한 한국어 응답 유도

### 3. 사용자 인터페이스
- 대화형 콘솔 인터페이스
- 간단한 명령어 지원 (q/quit, clear)
- 오류 처리 및 사용자 피드백

## 설치 방법

### 1. 사전 요구사항
- Python 3.7 이상
- Docker 및 Docker Compose
- Ollama 서비스 실행 중

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. Ollama 서비스 실행 확인
```bash
curl http://localhost:11434/api/tags
```

## 사용 방법

### 1. 애플리케이션 실행
```bash
python ollama_app.py
```

### 2. 대화 명령어
- 일반 대화: 원하는 메시지 입력
- 대화 종료: 'q' 또는 'quit' 입력
- 대화 기록 초기화: 'clear' 입력

## 코드 구조 설명

### 1. GemmaChatBot 클래스
```python
class GemmaChatBot:
    def __init__(self):
        # 모델 초기화
        self.chat_model = ChatOllama(
            model="gemma:3b",
            base_url="http://localhost:11434",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.7,
            streaming=True
        )
```
- ChatOllama 모델 초기화
- 스트리밍 출력 설정
- 온도(temperature) 파라미터로 응답 다양성 조절

### 2. 대화 처리
```python
def chat(self, user_input: str) -> str:
    # 메시지 생성 및 응답 처리
    messages = [self.system_prompt]
    if "history" in history:
        messages.extend(history["history"])
    messages.append(HumanMessage(content=user_input))
```
- 시스템 프롬프트 적용
- 대화 기록 관리
- 사용자 입력 처리

### 3. 메모리 관리
```python
self.memory = ConversationBufferMemory()
```
- ConversationBufferMemory를 사용한 대화 기록 저장
- 문맥을 고려한 응답 생성

## 주요 설정 파라미터

### 1. 모델 설정
- model: "gemma:3b"
- temperature: 0.7 (응답 다양성)
- streaming: True (실시간 출력)

### 2. 시스템 프롬프트
```python
self.system_prompt = SystemMessage(
    content="당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 응답해주세요."
)
```

## 성능 최적화

1. 스트리밍 응답
   - 실시간 응답 출력으로 사용자 경험 향상
   - 메모리 사용량 최적화

2. 오류 처리
   - 예외 상황 처리 및 사용자 피드백
   - 안정적인 실행 보장

## 주의사항

1. Ollama 서비스
   - Ollama 서비스가 실행 중이어야 함
   - localhost:11434 포트 접근 가능해야 함

2. 리소스 사용
   - Gemma 3B 모델 사용으로 인한 메모리 사용량 고려
   - 적절한 시스템 리소스 확보 필요

3. 네트워크 연결
   - 안정적인 네트워크 연결 필요
   - 초기 모델 로딩 시 시간 소요 가능

## 향후 개선 사항

1. 기능 추가
   - 다중 사용자 지원
   - 대화 내용 저장 및 로드
   - 고급 프롬프트 설정

2. 성능 개선
   - 응답 속도 최적화
   - 메모리 사용량 최적화
   - 더 나은 오류 처리

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 