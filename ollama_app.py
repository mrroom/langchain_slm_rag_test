from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
import warnings
warnings.filterwarnings('ignore')

class GemmaChatBot:
    def __init__(self):
        print("채팅 모델을 초기화하는 중입니다...")
        
        try:
            # Ollama 모델 설정
            self.chat_model = ChatOllama(
                model="gemma3:1b",
                base_url="http://localhost:11434",
                callbacks=[StreamingStdOutCallbackHandler()],
                temperature=0.7,
                streaming=True
            )
            
            # 대화 기록 설정
            self.memory = ConversationBufferMemory()
            
            # 시스템 프롬프트 설정
            self.system_prompt = SystemMessage(
                content="당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 응답해주세요."
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
            
            # 대화 기록 로드
            history = self.memory.load_memory_variables({})
            
            # 메시지 생성
            messages = [self.system_prompt]
            if "history" in history:
                messages.extend(history["history"])
            messages.append(HumanMessage(content=user_input))
            
            # 응답 생성
            print("\n챗봇: ", end='')
            response = self.chat_model.invoke(messages)
            
            # 대화 기록 저장
            self.memory.save_context(
                {"input": user_input},
                {"output": response.content}
            )
            
            return response.content
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

    def clear_memory(self):
        """대화 기록 초기화"""
        self.memory.clear()
        print("대화 기록이 초기화되었습니다.")

def main():
    print("Gemma 채팅 시스템을 시작합니다...")
    
    try:
        chatbot = GemmaChatBot()
        
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
