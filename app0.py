# 한국어 경량 모델 기반 RAG 시스템: LangChain + pgvector + Transformers
import os
from typing import List

# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

# 필요한 임포트
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import gc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# PostgreSQL 연결 정보 설정
CONNECTION_STRING = os.environ.get("PG_CONNECTION_STRING", "postgresql://postgres:swbang2413@10.41.0.152:5432/postgres")
COLLECTION_NAME = "korean_document_collection"

# 토큰 없이 다운로드 가능한 경량 한국어 모델
MODEL_NAME = "monologg/distilkobert"  # 경량화된 한국어 BERT
SENTENCE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"  # 경량 다국어 임베딩

class KoreanLanguageModel:
    def __init__(self, max_length=64):
        # 더 가벼운 모델 선택
        self.MODEL_NAME = "snunlp/KoSimCSE-roberta-small"  # 더 작은 모델로 변경
        self.SENTENCE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"
        self.max_length = max_length
        
        print("모델 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.sentence_model = SentenceTransformer(self.SENTENCE_MODEL_NAME)
        print("모델 로딩 완료!")
        
        # CPU 모드 설정
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)
    
    def get_word_embeddings(self, text):
        """단어/문장 임베딩 및 결과 처리"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings = outputs.last_hidden_state.mean(dim=1)
            numpy_embeddings = embeddings.cpu().numpy()
            normalized_embeddings = numpy_embeddings / np.linalg.norm(numpy_embeddings, axis=1, keepdims=True)
            
            self.clean_memory()
            
            return {
                'embeddings': normalized_embeddings,
                'dimension': normalized_embeddings.shape[1],
                'text': text
            }
        except Exception as e:
            print(f"임베딩 생성 중 오류 발생: {str(e)}")
            return None
    
    def get_sentence_embeddings(self, texts, batch_size=2):
        """문장 임베딩 및 결과 처리 (배치 처리)"""
        try:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                embeddings = self.sentence_model.encode(batch_texts)
                all_embeddings.append(embeddings)
                self.clean_memory()
            
            embeddings = np.vstack(all_embeddings)
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return {
                'embeddings': normalized_embeddings,
                'dimension': normalized_embeddings.shape[1],
                'texts': texts
            }
        except Exception as e:
            print(f"문장 임베딩 생성 중 오류 발생: {str(e)}")
            return None
    
    def find_similar_sentences(self, query, reference_texts, top_k=3):
        """유사 문장 찾기"""
        try:
            query_embedding = self.sentence_model.encode([query])[0]
            reference_embeddings = self.sentence_model.encode(reference_texts)
            
            similarities = cosine_similarity([query_embedding], reference_embeddings)[0]
            top_k_idx = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_k_idx:
                results.append({
                    'text': reference_texts[idx],
                    'similarity': float(similarities[idx])
                })
            
            self.clean_memory()
            return results
            
        except Exception as e:
            print(f"유사 문장 검색 중 오류 발생: {str(e)}")
            return None
    
    def clean_memory(self):
        """메모리 정리"""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def print_embedding_results(result_dict, result_type="단어"):
    print(f"\n=== {result_type} 임베딩 결과 ===")
    print(f"임베딩 차원: {result_dict['dimension']}")
    if result_type == "단어":
        print(f"입력 텍스트: {result_dict['text']}")
    else:
        print(f"입력 텍스트 수: {len(result_dict['texts'])}")
    print(f"임베딩 형태: {result_dict['embeddings'].shape}")
    print("임베딩 샘플 (처음 5개 값):", result_dict['embeddings'][0][:5])

# 1. 문서 로딩 및 분할
def load_and_split_documents(file_path: str) -> List[Document]:
    # 단일 파일인 경우
    if os.path.isfile(file_path):
        loader = TextLoader(file_path, encoding="utf-8")  # 한글 인코딩 지정
        documents = loader.load()
    # 디렉토리인 경우
    elif os.path.isdir(file_path):
        loader = DirectoryLoader(file_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        documents = loader.load()
    else:
        raise ValueError(f"경로가 유효하지 않습니다: {file_path}")
    
    # 한글 문서에 최적화된 분할 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # 한글은 영어보다 정보 밀도가 높으므로 청크 크기 조정
        chunk_overlap=150,   # 적절한 오버랩 설정
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # 한글 문서 구분자
    )
    
    return text_splitter.split_documents(documents)

# 2. 한국어 임베딩 모델 초기화
def initialize_korean_embeddings():
    # 한국어에 최적화된 경량 임베딩 모델 사용
    return HuggingFaceEmbeddings(
        model_name="snunlp/KoSimCSE-roberta-small",  # 한국어 특화 SBERT 모델
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# 3. 벡터 스토어 초기화
def initialize_vector_store(documents: List[Document], embeddings):
    return PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )

# 4. 기존 벡터 스토어에 연결
def connect_to_vector_store(embeddings):
    return PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

# 5. 한국어 경량 LLM 모델 초기화
def initialize_korean_llm():
    # 경량 한국어 모델 선택 (beomi/KoAlpaca-5.8B 모델은 한국어에 최적화된 경량 모델)
    model_name = "beomi/KoAlpaca-5.8B"
    
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 모델 캐싱 디렉토리 설정
    cache_dir = "./model_cache"
    # 모델 초기화 (8bit 양자화로 메모리 사용량 감소)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True,  # 8bit 양자화로 메모리 효율 증가
        cache_dir=cache_dir
    )
    
    # 파이프라인 생성
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,    # 답변 길이 제한
        temperature=0.7,       # 다양성 조절
        top_p=0.92,            # 샘플링 조절
        repetition_penalty=1.15,  # 반복 방지
        do_sample=True,
    )
    
    # LangChain에서 사용할 수 있는 형태로 변환
    return HuggingFacePipeline(pipeline=pipe)

# 6. RAG 체인 구성 (한국어 프롬프트 사용)
def create_qa_chain(vector_store, llm):
    # 한국어 프롬프트 템플릿 구성
    template = """
    다음 정보를 참고하여 질문에 상세히 답변해주세요:
    
    {context}
    
    질문: {question}
    
    답변:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 검색된 문서를 하나의 컨텍스트로 합침
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # 상위 3개 문서 검색
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# 7. 질문에 답변하는 함수
def answer_question(qa_chain, query: str):
    result = qa_chain({"query": query})
    
    # 결과 반환
    return {
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }

# 8비트 양자화 적용
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True
)

# 또는 더 작은 배치 크기 사용
batch_size = 1

# 메인 함수
def main():
    try:
        print("모델 초기화 중...")
        klm = KoreanLanguageModel(max_length=64)
        
        test_texts = [
            "오늘 날씨가 참 좋네요",
            "내일은 비가 올 것 같아요",
            "주말에 등산을 가려고 합니다"
        ]
        
        print("\n단어 임베딩 테스트 중...")
        word_emb = klm.get_word_embeddings(test_texts[0])
        if word_emb:
            print("\n=== 단어 임베딩 결과 ===")
            print(f"차원: {word_emb['dimension']}")
            print(f"입력: {word_emb['text']}")
            print(f"임베딩 샘플: {word_emb['embeddings'][0][:5]}")
        
        print("\n문장 임베딩 테스트 중...")
        sent_emb = klm.get_sentence_embeddings(test_texts, batch_size=2)
        if sent_emb:
            print("\n=== 문장 임베딩 결과 ===")
            print(f"차원: {sent_emb['dimension']}")
            print(f"문장 수: {len(sent_emb['texts'])}")
            print(f"임베딩 샘플: {sent_emb['embeddings'][0][:5]}")
        
        print("\n유사도 테스트 중...")
        query = "날씨가 좋아서 산책을 가려고 해요"
        similar = klm.find_similar_sentences(query, test_texts)
        if similar:
            print("\n=== 유사 문장 검색 결과 ===")
            print(f"검색어: {query}")
            for i, result in enumerate(similar, 1):
                print(f"\n{i}위: {result['text']}")
                print(f"유사도: {result['similarity']:.4f}")
        
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")
    finally:
        print("\n메모리 정리 중...")
        klm.clean_memory()
        print("프로그램 종료")

if __name__ == "__main__":
    main()