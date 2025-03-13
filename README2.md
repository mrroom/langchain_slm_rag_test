# 다중 문서 형식 벡터 검색 시스템

이 프로젝트는 다양한 형식(TXT, PDF, DOCX)의 문서를 처리하고 벡터 데이터베이스에 저장하여 의미 기반 검색을 수행하는 시스템입니다.

## 주요 기능

### 1. 다중 문서 형식 지원
- 텍스트 파일 (.txt)
- PDF 문서 (.pdf)
- Word 문서 (.docx)
- 확장 가능한 로더 구조

### 2. 자동 문서 처리
- 문서 자동 로드 및 분할
- 메타데이터 자동 추출
- 청크 단위 처리
- 에러 복원력

### 3. 벡터 검색
- 한국어 특화 임베딩
- 유사도 기반 검색
- 점수 기반 필터링
- 메타데이터 포함 결과

## 시스템 구조

### 1. DocumentProcessor 클래스
```python
class DocumentProcessor:
    def __init__(self, collection_name: str)
    def load_documents(self, directory: str) -> List[Document]
    def process_documents(self, documents: List[Document]) -> List[Document]
    def add_to_vectorstore(self, documents: List[Document]) -> int
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]
```

### 2. 주요 컴포넌트
- 임베딩 모델: `jhgan/ko-sroberta-multitask`
- 벡터 저장소: PostgreSQL + pgvector
- 문서 로더: langchain_community.document_loaders
- 텍스트 분할: CharacterTextSplitter

## 설치 방법

### 1. 시스템 요구사항
```bash
# Ubuntu/Debian 시스템 패키지
sudo apt-get update
sudo apt-get install -y python3-dev libpython3-dev
sudo apt-get install -y postgresql postgresql-contrib
sudo apt-get install -y pandoc
```

### 2. Python 패키지
```bash
pip install -r requirements.txt
```

### 3. requirements.txt
```text
langchain>=0.1.0
langchain-community>=0.0.1
psycopg2-binary>=2.9.9
pgvector>=0.2.4
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.2
unstructured>=0.10.30
unstructured[all-docs]>=0.10.30
pypdf>=3.0.0
python-magic>=0.4.27
```

## 사용 방법

### 1. 디렉토리 구조 