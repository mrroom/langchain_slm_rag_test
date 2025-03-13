from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Any
import glob

# PostgreSQL 연결 정보
CONNECTION_STRING = "postgresql+psycopg2://postgres:swbang2413@localhost:5432/postgres"

class DocumentProcessor:
    
    def __init__(self, collection_name: str):
        self.COLLECTION_NAME = collection_name
        
        # 한국어 임베딩 모델 초기화
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 8, 'normalize_embeddings': True}
        )
        
        # PGVector 초기화
        try:
            self.vector_store = PGVector(
                connection_string=CONNECTION_STRING,
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings
            )
            print(f"컬렉션 '{self.COLLECTION_NAME}' 연결 성공")
        except Exception as e:
            print(f"데이터베이스 연결 오류: {str(e)}")
            raise

    def load_documents(self, directory: str) -> List[Document]:
        """다양한 형식의 문서 로드"""
        documents = []
        
        # 지원하는 파일 형식과 해당 로더
        loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': UnstructuredWordDocumentLoader
        }
        
        try:
            # 디렉토리 내의 모든 파일 검색
            for ext, loader_class in loaders.items():
                files = glob.glob(os.path.join(directory, f'*{ext}'))
                for file_path in files:
                    try:
                        print(f"파일 로드 시도: {file_path}")
                        loader = loader_class(file_path)
                        file_documents = loader.load()
                        
                        # 메타데이터 추가
                        for doc in file_documents:
                            doc.metadata.update({
                                'source': file_path,
                                'filename': os.path.basename(file_path),
                                'file_type': ext,
                                'load_time': datetime.now().isoformat()
                            })
                        documents.extend(file_documents)
                        print(f"로드 완료: {file_path}")
                    except Exception as e:
                        print(f"파일 로드 실패 {file_path}: {str(e)}")
                        continue
            
            return documents
        except Exception as e:
            print(f"문서 로드 중 오류 발생: {str(e)}")
            return []

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """문서 전처리 및 청크 분할"""
        try:
            if not documents:
                return []
                
            # 텍스트 분할기 설정
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            
            # 문서 분할
            split_docs = text_splitter.split_documents(documents)
            print(f"총 {len(split_docs)}개의 청크로 분할됨")
            return split_docs
            
        except Exception as e:
            print(f"문서 처리 중 오류 발생: {str(e)}")
            return []

    def add_to_vectorstore(self, documents: List[Document]) -> int:
        """벡터 저장소에 문서 추가"""
        try:
            if not documents:
                print("처리할 문서가 없습니다.")
                return 0
                
            self.vector_store.add_documents(documents)
            return len(documents)
            
        except Exception as e:
            print(f"벡터 저장소 추가 중 오류 발생: {str(e)}")
            return 0

    def search(self, query: str, k: int = 3, score_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """문서 검색"""
        try:
            # 유사도 검색 수행
            results_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=k*2
            )
            
            # 결과 필터링 및 정렬
            filtered_results = []
            for doc, score in results_with_scores:
                similarity_score = 1.0 / (1.0 + score)
                if similarity_score >= score_threshold:
                    filtered_results.append({
                        'text': doc.page_content,
                        'metadata': doc.metadata,
                        'score': similarity_score
                    })
            
            # 점수순 정렬
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            return filtered_results[:k]
            
        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")
            return []

def main():
    try:
        # 문서 처리기 초기화
        collection_name = "korean_documents_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        processor = DocumentProcessor(collection_name)
        
        # 문서 로드
        print("\n문서 로드 중...")
        documents_dir = "./documents"  # 문서 디렉토리 경로
        if not os.path.exists(documents_dir):
            print(f"documents 디렉토리가 없습니다. 생성합니다: {documents_dir}")
            os.makedirs(documents_dir)
            
        documents = processor.load_documents(documents_dir)
        print(f"총 {len(documents)}개의 문서 로드됨")
        
        # 문서 처리 및 저장
        if documents:
            print("\n문서 처리 중...")
            processed_docs = processor.process_documents(documents)
            
            print("\n벡터 저장소에 추가 중...")
            added_count = processor.add_to_vectorstore(processed_docs)
            print(f"총 {added_count}개의 청크가 저장됨")
            
            # 검색 테스트
            print("\n검색 테스트:")
            test_queries = [
                "인공지능 기술",
                "데이터 분석 방법",
                "프로젝트 관리"
            ]
            
            for query in test_queries:
                print(f"\n[검색어]: {query}")
                results = processor.search(query, k=3)
                
                print("\n[검색 결과]:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. 문서 출처: {result['metadata']['filename']}")
                    print(f"   텍스트: {result['text'][:200]}...")
                    print(f"   유사도 점수: {result['score']:.4f}")
        else:
            print("처리할 문서가 없습니다. documents 폴더에 문서를 추가해주세요.")
        
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
