# 라즈베리파이용 한국어 챗봇 시스템

이 프로젝트는 라즈베리파이 환경에 최적화된 경량 한국어 챗봇 시스템입니다. LangChain과 HuggingFace의 Transformers를 활용하여 구현되었습니다.

## 주요 기능

### 1. 경량 한국어 모델
- EleutherAI/polyglot-ko-1.3b 모델 사용
- 라즈베리파이 환경에 최적화
- 토큰 없이 사용 가능한 오픈 소스 모델

### 2. 메모리 최적화
- 저용량 CPU 메모리 사용 설정
- 자동 디바이스 매핑
- 효율적인 토큰 관리

### 3. 대화 기능
- 대화 기록 유지
- 스트리밍 방식의 응답 출력
- 대화 초기화 기능

## 시스템 구조

### KoreanChatBot 클래스
```python
class KoreanChatBot:
    def __init__(self)      # 모델 초기화
    def chat(self)          # 대화 처리
    def clear_memory()      # 메모리 초기화
```

## 설치 방법

### 1. 시스템 요구사항
- Python 3.7+
- 최소 2GB RAM
- Raspberry Pi 4 권장

### 2. 필요한 패키지
```text
langchain>=0.1.0
transformers>=4.30.0
torch>=2.0.0
```

### 3. 설치 명령어
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 1. 기본 실행
```bash
python chat_app.py
```

### 2. 대화 명령어
- `q` 또는 `quit`: 프로그램 종료
- `clear`: 대화 기록 초기화

### 3. 대화 예시 