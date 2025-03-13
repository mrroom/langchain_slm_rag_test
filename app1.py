from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import numpy as np
import os

# PostgreSQL 연결 정보
CONNECTION_STRING = "postgresql+psycopg2://postgres:swbang2413@localhost:5432/postgres"
COLLECTION_NAME = "korean_documents"

class KoreanVectorDB:

    # 한국어 임베딩 모델 초기화
    def __init__(self):
        # 한국어 임베딩 모델 초기화 (수정된 부분)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={
                'device': 'cpu'
            },
            encode_kwargs={
                'batch_size': 8,
                'normalize_embeddings': True
            }
        )
        
        # PGVector 초기화
        try:
            self.vector_store = PGVector(
                connection_string=CONNECTION_STRING,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings
            )
            print("벡터 데이터베이스 연결 성공")
        except Exception as e:
            print(f"데이터베이스 연결 오류: {str(e)}")
            raise

    # 텍스트 처리
    def preprocess_text(self, text):
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
        # 기본적인 텍스트 정리
        text = text.strip()
        # 중복 공백 제거
        text = ' '.join(text.split())
        return text

    # 문서 생성
    def create_documents(self, texts):
        """문서 생성 with 메타데이터"""
        documents = []
        for i, text in enumerate(texts):
            # 텍스트 전처리
            processed_text = self.preprocess_text(text)
            if not processed_text:  # 빈 문자열 건너뛰기
                continue
                
            # 메타데이터 추가
            metadata = {
                'id': str(i),
                'length': len(processed_text),
                'source': 'user_input'
            }
            
            doc = Document(
                page_content=processed_text,
                metadata=metadata
            )
            documents.append(doc)
        return documents

    # 문서 추가
    def add_documents(self, texts):
        """문서 추가 with 개선된 처리"""
        try:
            # 문서 생성
            documents = self.create_documents(texts)
            if not documents:
                print("처리할 문서가 없습니다.")
                return 0
                
            # 텍스트 분할
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=200,
                chunk_overlap=30,
                length_function=len
            )
            split_docs = text_splitter.split_documents(documents)
            
            # 벡터 저장소에 추가
            self.vector_store.add_documents(split_docs)
            print(f"{len(split_docs)}개의 문서 청크가 추가되었습니다.")
            return len(split_docs)
        except Exception as e:
            print(f"문서 추가 중 오류 발생: {str(e)}")
            raise

    # 검색
    def search(self, query, k=3, score_threshold=0.2):
        """개선된 유사 문서 검색"""
        try:
            # 쿼리 전처리
            processed_query = self.preprocess_text(query)
            if not processed_query:
                return []
                
            # 유사도 검색 수행
            results_with_scores = self.vector_store.similarity_search_with_score(
                processed_query, 
                k=k*2  # 더 많은 후보 검색
            )
            
            # 결과 필터링 및 정렬
            filtered_results = []
            for doc, score in results_with_scores:
                similarity_score = 1.0 / (1.0 + score)  # 거리를 유사도 점수로 변환
                if similarity_score >= score_threshold:
                    filtered_results.append({
                        'document': doc,
                        'score': similarity_score,
                        'text': doc.page_content,
                        'metadata': doc.metadata
                    })
            
            # 점수로 정렬하고 상위 k개만 선택
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            return filtered_results[:k]
            
        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")
            raise

def main():
    # 테스트 데이터
    sample_texts = [
        "오늘 날씨가 매우 좋아서 공원에서 산책을 했습니다. 햇살이 따뜻했어요.",
        "주말에는 가족들과 함께 등산을 가기로 했어요. 준비물을 미리 챙겨놓았습니다.",
        "비가 와서 실내에서 책을 읽으며 시간을 보냈습니다. 차 한잔과 함께 여유롭게 보냈어요.",
        "새로 개봉한 영화를 친구들과 함께 보러 갔습니다. 재미있는 시간이었어요.",
        "맛있는 저녁 식사를 준비하며 하루를 마무리했어요. 오늘 요리한 음식이 특별히 맛있었습니다.",
        "아침 일찍 일어나서 요가를 했어요. 상쾌한 아침 운동이었습니다.",
        "집 근처 공원에서 조깅을 했습니다. 날씨가 운동하기 좋았어요.",
        "비가 오는 날씨에 실내 카페에서 음악을 들으며 휴식을 취했습니다."
    ]

    try:
        print("벡터 데이터베이스 초기화 중...")
        db = KoreanVectorDB()

        #print("\n문서 추가 중...")
        #num_chunks = db.add_documents(sample_texts)
        #print(f"총 {num_chunks}개의 청크가 처리되었습니다.")

        print("\n검색 테스트:")
        queries = [
            "날씨가 좋은 날 야외 활동",
            "비오는 날 실내 활동",
            "가족과 함께하는 활동"
        ]

        for query in queries:
            print(f"\n[검색어]: {query}")
            results = db.search(query, k=3, score_threshold=0.2)
            
            print("\n[검색 결과]:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. 문서: {result['text']}")
                print(f"   유사도 점수: {result['score']:.4f}")
                print(f"   메타데이터: {result['metadata']}")
                if result['score'] > 0.6:
                    print(f"\n{i}. 선택 문서: {result['text']}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
