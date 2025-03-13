from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

class KoreanChatBot:
    def __init__(self):
        print("채팅 모델을 초기화하는 중입니다...")
        
        # 경량 한국어 모델 설정 (변경됨)
        model_name = "EleutherAI/polyglot-ko-1.3b"  # 더 작은 모델로 변경
        
        try:
            # 토크나이저 로드
            print("토크나이저 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # 모델 로드
            print("모델 로드 중...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map='auto',
                trust_remote_code=True
            )
            
            # 파이프라인 설정
            print("파이프라인 설정 중...")
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=128,  # 토큰 수 감소
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Langchain 설정
            self.llm = HuggingFacePipeline(
                pipeline=pipe,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            
            # 대화 메모리 설정
            self.memory = ConversationBufferMemory()
            
            # 대화 체인 설정
            self.conversation = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                verbose=False
            )
            
            print("모델 초기화가 완료되었습니다!")
            
        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {str(e)}")
            raise

    def chat(self, user_input: str) -> str:
        """사용자 입력에 대한 응답 생성"""
        try:
            if not user_input.strip():
                return "입력을 확인해주세요."
            
            # 프롬프트 형식화
            formatted_input = f"사용자: {user_input}\n시스템: "
            
            response = self.conversation.predict(input=formatted_input)
            return response.strip()
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

    def clear_memory(self):
        """대화 기록 초기화"""
        self.memory.clear()
        print("대화 기록이 초기화되었습니다.")

def main():
    print("한국어 채팅 시스템을 시작합니다...")
    
    try:
        chatbot = KoreanChatBot()
        
        print("\n대화를 시작합니다! (종료하려면 'q' 또는 'quit'를 입력하세요)")
        print("대화 기록을 초기화하려면 'clear'를 입력하세요")
        
        while True:
            user_input = input("\n사용자: ").strip()
            
            if user_input.lower() in ['q', 'quit']:
                print("\n대화를 종료합니다.")
                break
                
            if user_input.lower() == 'clear':
                chatbot.clear_memory()
                continue
                
            if user_input:
                print("\n챗봇: ", end='')
                response = chatbot.chat(user_input)
                if not response:
                    print("응답이 생성되지 않았습니다.")
            else:
                print("입력을 확인해주세요.")
                
    except Exception as e:
        print(f"\n시스템 오류 발생: {str(e)}")
        print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main() 