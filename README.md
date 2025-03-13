# 한국어 자연어 처리 응용 시스템

이 프로젝트는 라즈베리파이 환경에 최적화된 두 가지 한국어 자연어 처리 시스템을 제공합니다.

## 시스템 비교

### 1. RAG(Retrieval-Augmented Generation) 시스템
[상세 설명은 README_RAG_APP.md 참조]

- **주요 특징**
  - 다중 문서 형식 지원 (TXT, PDF, DOCX)
  - 벡터 데이터베이스 기반 검색
  - 의미 기반 문서 검색

- **핵심 기능**
  - 문서 자동 처리 및 임베딩
  - 효율적인 벡터 검색
  - PostgreSQL + pgvector 활용

- **사용 사례**
  - 문서 관리 및 검색
  - 정보 검색 시스템
  - 지식 베이스 구축

### 2. 챗봇 시스템
[상세 설명은 README_CHAT_APP.md 참조]

- **주요 특징**
  - 경량 한국어 언어 모델 사용
  - 대화형 인터페이스
  - 메모리 최적화

- **핵심 기능**
  - 실시간 대화 처리
  - 대화 기록 관리
  - 스트리밍 응답

- **사용 사례**
  - 대화형 어시스턴트
  - 간단한 질의응답
  - 챗봇 서비스

## 시스템 요구사항

### 공통 요구사항
- Python 3.7+
- Raspberry Pi 4
- 최소 2GB RAM

### 필수 패키지
```text
langchain>=0.1.0
transformers>=4.30.0
torch>=2.0.0
```

## 빠른 시작

### RAG 시스템
```bash
# 1. 데이터베이스 설정
psql -U postgres
CREATE DATABASE your_db_name;
\c your_db_name
CREATE EXTENSION vector;

# 2. 환경 설정
cp .env.example .env
# .env 파일 편집

# 3. 실행
python rag_app.py
```

### 챗봇 시스템
```bash
# 1. 가상환경 설정
python -m venv venv
source venv/bin/activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 실행
python chat_app.py
```

## 시스템 선택 가이드

### RAG 시스템 선택 시기
- 문서 기반 검색이 필요할 때
- 정확한 정보 검색이 중요할 때
- 다양한 문서 형식을 처리해야 할 때

### 챗봇 시스템 선택 시기
- 대화형 인터페이스가 필요할 때
- 일반적인 대화 처리가 필요할 때
- 리소스가 제한적일 때

## 프로젝트 구조
```
project/
├── rag_app.py            # RAG 시스템
├── chat_app.py           # 챗봇 시스템
├── .env                  # 환경 설정
├── README.md            # 본 문서
├── README_RAG_APP.md    # RAG 시스템 상세 설명
├── README_CHAT_APP.md   # 챗봇 시스템 상세 설명
└── requirements.txt     # 패키지 요구사항
```

## 상세 문서
- [RAG 시스템 상세 설명](README_RAG_APP.md)
- [챗봇 시스템 상세 설명](README_CHAT_APP.md)

## 라이선스
MIT License
