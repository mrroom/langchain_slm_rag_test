한글에 최적화된 경량 모델을 사용하여 RAG 시스템을 구현하겠습니다. 한국어에 잘 작동하는 경량 SLM(Small Language Model)과 임베딩 모델을 선택하겠습니다.

```python
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

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# PostgreSQL 연결 정보 설정
CONNECTION_STRING = os.environ.get("PG_CONNECTION_STRING", "postgresql://username:password@localhost:5432/vectordb")
COLLECTION_NAME = "korean_document_collection"

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
        model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",  # 한국어 특화 SBERT 모델
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
    # 경량 한국어 모델 선택 (beomi/KoAlpaca-6B4b 모델은 한국어에 최적화된 경량 모델)
    model_name = "beomi/KoAlpaca-6B4b-16K"
    
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 모델 초기화 (8bit 양자화로 메모리 사용량 감소)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True,  # 8bit 양자화로 메모리 효율 증가
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

# 메인 함수
def main():
    # 1. 문서 로딩 및 분할
    print("문서를 로딩하고 분할하는 중...")
    documents = load_and_split_documents("./documents")  # 문서가 저장된 폴더 경로
    
    # 2. 한국어 임베딩 모델 초기화
    print("한국어 임베딩 모델을 초기화하는 중...")
    embeddings = initialize_korean_embeddings()
    
    # 3. 벡터 스토어 초기화 또는 연결
    try:
        print("기존 벡터 스토어에 연결 시도 중...")
        vector_store = connect_to_vector_store(embeddings)
    except:
        print("새 벡터 스토어를 생성하는 중...")
        vector_store = initialize_vector_store(documents, embeddings)
    
    # 4. 한국어 LLM 초기화
    print("한국어 LLM 모델을 초기화하는 중...")
    llm = initialize_korean_llm()
    
    # 5. RAG 체인 생성
    print("RAG 체인을 구성하는 중...")
    qa_chain = create_qa_chain(vector_store, llm)
    
    # 6. 질문 응답
    while True:
        query = input("\n질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if query.lower() == "exit":
            break
            
        print("답변을 생성하는 중...")
        result = answer_question(qa_chain, query)
        
        print("\n답변:")
        print(result["answer"])
        
        print("\n참고 문서:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"문서 {i+1}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    main()

```

```text
# RAG 시스템 필수 패키지
langchain==0.1.9
langchain-community==0.0.20
psycopg2-binary==2.9.9
pgvector==0.2.5
transformers==4.39.0
torch==2.1.2
numpy==1.26.3
python-dotenv==1.0.1
sentence-transformers==2.4.0
accelerate==0.27.2
bitsandbytes==0.42.0
protobuf==4.25.1
safetensors==0.4.2
einops==0.7.0

```

이 코드는 한국어에 최적화된 경량 모델을 사용하여 RAG 시스템을 구현했습니다. 주요 특징은 다음과 같습니다:

### 한국어 최적화 모델 선택:
1. **임베딩 모델**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS` - 한국어에 특화된 SBERT 모델로, 한글 문서의 의미를 효과적으로 벡터화합니다.
2. **언어 모델**: `beomi/KoAlpaca-6B4b-16K` - 한국어에 최적화된 알파카 모델로, 8bit 양자화를 적용하여 메모리 사용량을 줄였습니다.

### 한국어 처리를 위한 특화 기능:
1. **UTF-8 인코딩** 사용하여 한글 문서 로드
2. **한글 문서에 맞게 조정된 청크 크기와 구분자** 설정
3. **한국어 프롬프트 템플릿** 사용

### 시스템 설치 및 실행 방법:
1. `requirements.txt` 파일을 사용하여 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```
2. PostgreSQL에 pgvector 확장이 설치되어 있어야 합니다.
3. `.env` 파일에 데이터베이스 연결 정보 설정:
   ```
   PG_CONNECTION_STRING=postgresql://username:password@localhost:5432/vectordb
   ```
4. 문서를 `./documents` 폴더에 넣고 코드 실행:
   ```bash
   python rag_system.py
   ```

이 시스템은 한국어 문서를 효율적으로 처리하고, 비교적 적은 리소스로 한국어 질의응답이 가능한 RAG 시스템을 구현했습니다. 필요에 따라 모델 크기나 매개변수를 조정하여 성능과 리소스 사용량의 균형을 맞출 수 있습니다.

네, 물론입니다! 테스트를 위한 샘플 한국어 문서를 몇 개 만들어 드리겠습니다.

```text
# 인공지능의 개요와 발전 과정

인공지능(AI)은 인간의 학습능력과 추론능력, 지각능력, 자연언어의 이해능력 등을 컴퓨터 프로그램으로 실현한 기술입니다. 인공지능이라는 용어는 1956년 다트머스 회의에서 존 매카시에 의해 처음 제안되었습니다.

## 인공지능의 주요 분야

1. **머신러닝(Machine Learning)**: 데이터를 기반으로 패턴을 학습하고 예측하는 알고리즘입니다. 지도학습, 비지도학습, 강화학습으로 나눌 수 있습니다.

2. **딥러닝(Deep Learning)**: 인간의 뇌 구조를 모방한 인공신경망을 기반으로 하는 머신러닝의 한 분야입니다. 이미지 인식, 자연어 처리 등에서 혁신적인 성과를 보이고 있습니다.

3. **자연어 처리(NLP)**: 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다. 번역, 감성 분석, 정보 추출 등의 작업을 수행합니다.

4. **컴퓨터 비전(Computer Vision)**: 컴퓨터가 이미지나 비디오를 이해하고 처리하는 기술입니다. 얼굴 인식, 객체 탐지 등에 활용됩니다.

## 한국의 인공지능 연구

한국은 2016년 알파고와 이세돌의 바둑 대결 이후 인공지능에 대한 관심이 크게 증가했습니다. 정부 차원의 인공지능 국가전략을 수립하고, 다양한 산업 분야에서 인공지능 기술을 접목하려는 노력이 진행 중입니다.

주요 연구 기관으로는 KAIST AI대학원, 서울대학교 인공지능 연구소, 네이버 랩스, 카카오브레인 등이 있으며, 삼성전자, LG전자, 현대자동차 등 대기업들도 인공지능 기술 개발에 많은 투자를 하고 있습니다.

## 인공지능의 윤리적 문제

인공지능의 발전과 함께 윤리적 문제도 대두되고 있습니다. 알고리즘 편향성, 프라이버시 침해, 자율주행차의 윤리적 의사결정, 일자리 대체 문제 등 다양한 사회적 이슈들이 논의되고 있으며, 이를 해결하기 위한 윤리적 가이드라인과 법적 규제의 필요성이 강조되고 있습니다.

향후 인공지능은 인간의 삶을 더욱 편리하게 만들어줄 것으로 기대되지만, 기술의 발전과 함께 사회적 합의와 윤리적 고려가 병행되어야 할 것입니다.

```

```text
# 한국의 역사: 고대에서 현대까지

## 고대 시대 (기원전 - 668년)

한반도의 최초 국가로 알려진 고조선은 기원전 2333년에 단군왕검에 의해 건국되었다고 전해집니다. 이후 삼국시대가 전개되었으며, 고구려, 백제, 신라가 한반도를 분할하여 지배했습니다.

고구려는 만주와 한반도 북부 지역에 걸친 광활한 영토를 다스렸으며, 강력한 군사력을 바탕으로 중국 왕조들과 대립했습니다. 백제는 한반도 서남부를 지배하며 해상 무역과 예술 문화가 발달했습니다. 신라는 한반도 남동부를 통치하다가 668년 삼국을 통일하였습니다.

## 통일신라와 고려시대 (668년 - 1392년)

통일신라는 불교 문화의 황금기를 맞이했으며, 불국사와 석굴암 같은 문화유산을 남겼습니다. 후삼국 시대를 거쳐 918년 고려가 건국되었습니다. 고려는 문화와 예술이 크게 발전한 시기로, 금속활자 인쇄술을 발명하고 고려청자를 제작했습니다. 또한 무신정권과 몽골의 침략과 같은 외부 도전도 겪었습니다.

## 조선시대 (1392년 - 1910년)

1392년 이성계가 고려를 무너뜨리고 조선을 건국했습니다. 조선은 유교를 국가 이념으로 삼고 한글을 창제하는 등 문화적 업적을 이루었습니다. 임진왜란과 병자호란 같은 외세의 침략을 겪기도 했습니다. 19세기 말에는 개항과 함께 서구 문물이 유입되었으며, 1910년 일본에 강제 병합되었습니다.

## 일제 강점기 (1910년 - 1945년)

일본의 식민지배 아래 한국인들은 민족 정체성을 지키기 위해 노력했습니다. 1919년 3.1 운동을 통해 독립 의지를 세계에 알렸으며, 임시정부를 수립하고 독립운동을 전개했습니다. 1945년 8월 15일 일본의 항복으로 해방을 맞이했습니다.

## 분단과 현대 (1945년 - 현재)

해방 이후 미국과 소련의 분할 점령으로 한반도는 남북으로 나뉘게 되었습니다. 1948년 대한민국 정부가 수립되었으며, 1950년 한국전쟁이 발발하여 1953년 휴전 상태로 마무리되었습니다.

1960년대부터 시작된 경제개발 계획을 통해 대한민국은 '한강의 기적'이라 불리는 경제 성장을 이루었습니다. 1987년 민주화 운동으로 민주주의 체제를 확립했으며, 21세기에는 K-팝, 영화 등 한류 문화가 세계적으로 인기를 얻으며 문화 강국으로 자리매김하고 있습니다.

현재 대한민국은 IT 기술, 반도체, 자동차, 조선 등 다양한 산업 분야에서 세계적인 경쟁력을 갖추고 있으며, 북한과의 평화 정착을 위한 노력도 지속하고 있습니다.

```

```text
# 한국의 음식 문화

한국 음식은 오랜 역사와 전통을 가진 문화유산으로, 다양한 식재료와 발효 기술을 바탕으로 발전해왔습니다. 한국 음식의 가장 큰 특징은 발효 식품의 다양성과 계절에 따른 식재료의 활용, 그리고 영양적 균형을 중시하는 식단 구성입니다.

## 주요 한국 음식

### 김치
김치는 한국을 대표하는 발효 식품으로, 배추를 비롯한 다양한 채소를 소금에 절인 후 고춧가루, 마늘, 생강 등의 양념과 함께 발효시킨 음식입니다. 2013년에는 유네스코 인류무형문화유산으로 등재되었습니다. 지역과 계절에 따라 200여 종 이상의 김치가 있으며, 대표적으로 배추김치, 깍두기, 총각김치, 동치미 등이 있습니다.

### 비빔밥
비빔밥은 밥 위에 다양한 나물과 고기, 계란 등을 올리고 고추장과 함께 비벼 먹는 음식입니다. 지역별로 전주 비빔밥, 진주 비빔밥, 해물 비빔밥 등 다양한 종류가 있습니다. 영양 균형이 뛰어나고 한 그릇에 다양한 재료를 섭취할 수 있어 건강식으로 세계적인 인기를 얻고 있습니다.

### 불고기
불고기는 얇게 썬 쇠고기를 간장, 설탕, 파, 마늘, 깨소금 등의 양념에 재운 후 구워 먹는 음식입니다. 달콤하고 짭짤한 맛으로 외국인들도 쉽게 접근할 수 있어 한국 음식 중 가장 널리 알려진 요리 중 하나입니다.

### 된장찌개
된장찌개는 대두를 발효시켜 만든 된장을 주재료로 한 찌개 요리입니다. 두부, 감자, 호박, 버섯, 양파 등 다양한 채소와 함께 끓여 깊은 맛을 냅니다. 한국인의 일상적인 식탁에서 빠지지 않는 음식 중 하나로, 단백질과 미네랄이 풍부한 건강식입니다.

## 한국 식문화의 특징

### 반찬 문화
한국 식사는 밥과 국, 그리고 여러 가지 반찬으로 구성됩니다. 김치, 나물, 생선, 고기 등 다양한 반찬이 한 상에 차려지며, 이를 통해 영양 균형을 맞추고 다양한 맛을 즐길 수 있습니다. 이러한 반찬 문화는 한국 음식의 가장 큰 특징 중 하나입니다.

### 발효 식품
한국 음식에서 발효 식품은 매우 중요한 위치를 차지합니다. 김치, 된장, 간장, 고추장 등의 발효 식품은 한국 요리의 기본이 되는 양념으로 사용됩니다. 발효 과정을 통해 독특한 맛과 향이 생기며, 영양소의 흡수율이 높아지고 프로바이오틱스가 생성되어 건강에도 이로운 효과가 있습니다.

### 약식동원(藥食同源)
한국 음식 문화에는 '약과 음식은 근원이 같다'는 약식동원의 철학이 바탕에 깔려 있습니다. 음식을 통해 건강을 유지하고 질병을 예방하려는 사상으로, 인삼, 마늘, 대추, 버섯 등 약용 식재료를 일상 식단에 적극적으로 활용합니다. 이러한 전통은 보양식 문화로 이어져 삼계탕, 엄나무 순대국 등 다양한 보양식이 발달했습니다.

## 현대 한국 음식의 세계화

최근 한류 열풍과 함께 한국 음식의 세계적 인기도 높아지고 있습니다. 비빔밥, 불고기, 김치 등 전통 한식뿐만 아니라 치킨, 떡볶이 등 현대적인 한국 음식도 세계인의 입맛을 사로잡고 있습니다. 특히 발효 식품에 대한 관심이 높아지면서 김치와 같은 한국의 발효 식품이 건강식으로 주목받고 있습니다.

정부와 민간 기업들은 한식의 세계화를 위해 다양한 노력을 기울이고 있으며, 해외 한식당의 수도 꾸준히 증가하고 있습니다. 또한 현지인의 입맛에 맞게 한식을 변형한 퓨전 한식도 인기를 얻고 있어, 앞으로 한국 음식 문화의 세계적 영향력은 더욱 확대될 것으로 기대됩니다.

```

```text
# 한국 주요 기업 현황 및 산업 동향

## 한국 경제와 기업의 성장

한국은 1960년대부터 급속한 경제 성장을 이루어 세계 10위권의 경제 규모를 갖추게 되었습니다. 이러한 성장의 중심에는 삼성, 현대, LG와 같은 대기업들이 있었으며, 이들은 한국 경제 발전의 견인차 역할을 했습니다. 한국 기업들은 전자, 자동차, 조선, 철강, 화학, 반도체 등 다양한 산업 분야에서 세계적인 경쟁력을 갖추고 있습니다.

## 주요 산업별 대표 기업

### 전자 및 반도체 산업
삼성전자는 한국을 대표하는 글로벌 기업으로, 스마트폰, TV, 가전제품 등 전자제품과 메모리 반도체 시장에서 세계적인 점유율을 보유하고 있습니다. 특히 메모리 반도체 분야에서는 세계 1위의 시장 점유율을 기록하고 있습니다.

SK하이닉스는 메모리 반도체 분야에 특화된 기업으로, DRAM과 NAND 플래시 메모리 시장에서 삼성전자와 함께 세계 시장을 주도하고 있습니다.

LG전자는 TV, 냉장고, 세탁기 등 가전제품 분야에서 강점을 보이며, OLED TV 시장에서 선두를 달리고 있습니다.

### 자동차 산업
현대자동차는 기아자동차와 함께 현대자동차그룹을 형성하여 세계 5위권의 자동차 제조사로 성장했습니다. 최근에는 전기차, 수소차 등 친환경 자동차 개발에 집중하며 미래 모빌리티 시장을 선도하려는 노력을 기울이고 있습니다.

### 조선 및 해운 산업
현대중공업그룹은 세계 최대 규모의 조선사로, 각종 선박과 해양플랜트 건설 분야에서 세계적인 경쟁력을 갖추고 있습니다. 대우조선해양, 삼성중공업과 함께 세계 조선 시장의 상위권을 차지하고 있습니다.

HMM(구 현대상선)은 한국을 대표하는 글로벌 해운회사로, 컨테이너 운송과 벌크화물 운송 서비스를 제공하고 있습니다.

### 화학 및 에너지 산업
SK이노베이션은 석유 정제, 석유화학, 윤활유, 전기차 배터리 등의 사업을 영위하고 있으며, 특히 전기차 배터리 분야에서 글로벌 기업으로 성장하고 있습니다.

LG화학은 석유화학, 전자재료, 생명과학, 전기차 배터리 등 다양한 분야에서 사업을 전개하고 있으며, 전기차 배터리 시장에서는 세계 선두권 기업 중 하나입니다.

롯데케미칼은 석유화학 분야에 특화된 기업으로, 에틸렌, 프로필렌 등 기초유분과 합성수지, 합성고무 등 다양한 화학제품을 생산하고 있습니다.

## 새로운 성장 동력 산업

### 바이오 및 제약 산업
삼성바이오로직스는 세계 최대 규모의 바이오의약품 위탁생산(CMO) 기업으로 성장했으며, 셀트리온과 함께 바이오시밀러 시장에서도 두각을 나타내고 있습니다.

SK바이오사이언스는 코로나19 백신 개발 및 생산에 참여하며 글로벌 백신 산업에서 존재감을 드러내고 있습니다.

### IT 및 서비스 산업
네이버는 국내 최대 포털 기업으로, 검색, 커머스, 핀테크, 콘텐츠 등 다양한 온라인 서비스를 제공하고 있습니다. 최근에는 인공지능, 클라우드, 로봇 등 미래 기술 분야에 투자를 확대하고 있습니다.

카카오는 모바일 메신저 '카카오톡'을 기반으로 콘텐츠, 커머스, 모빌리티, 핀테크 등 다양한 사업 영역으로 확장하고 있습니다.

쿠팡은 이커머스 시장에서 빠르게 성장하여 2021년 뉴욕 증시에 상장했으며, 물류 인프라를 기반으로 한 신속 배송 서비스로 경쟁력을 확보하고 있습니다.

## 미래 전망 및 도전 과제

한국 기업들은 4차 산업혁명 시대에 대응하기 위해 인공지능, 빅데이터, 로봇, 바이오 등 첨단 기술 분야에 대한 투자를 확대하고 있습니다. 또한 탄소중립 목표 달성을 위한 친환경 기술 개발과 ESG(환경·사회·지배구조) 경영 강화에도 노력을 기울이고 있습니다.

그러나 글로벌 경쟁 심화, 저출산·고령화로 인한 내수 시장 위축, 무역 분쟁과 보호무역주의 확산 등의 도전 요인도 존재합니다. 이러한 위기를 극복하고 지속 가능한 성장을 이루기 위해서는 혁신 역량 강화와 함께 새로운 성장 동력 발굴이 중요한 과제로 남아 있습니다.

```

이 네 개의 샘플 문서는 한국어 RAG 시스템을 테스트하기에 적합한 주제들로 구성했습니다:

1. **인공지능_개요.txt**: 인공지능의 기본 개념, 주요 분야, 한국의 인공지능 연구 현황, 윤리적 문제 등을 다룹니다.

2. **한국_역사_개요.txt**: 고대부터 현대까지 한국 역사의 중요한 시기와 사건들을 시대별로 정리했습니다.

3. **한국_음식_문화.txt**: 한국 전통 음식의 특징, 주요 대표 음식, 식문화의 특성, 세계화 현황 등을 소개합니다.

4. **한국_기업_현황.txt**: 한국의 주요 기업들과 산업 현황, 성장 분야, 미래 전망 등에 관한 정보를 담고 있습니다.

이 샘플 문서들을 사용하여 RAG 시스템을 테스트하면 다양한 주제에 대한 질의응답이 가능합니다. 예를 들어:

- "한국의 인공지능 연구 현황은 어떤가요?"
- "조선시대의 주요 업적은 무엇인가요?"
- "김치의 특징과 세계적 위상에 대해 알려주세요."
- "한국의 반도체 산업 경쟁력은 어떤가요?"

이러한 질문을 통해 시스템의 문서 검색 및 응답 생성 능력을 테스트할 수 있습니다.
