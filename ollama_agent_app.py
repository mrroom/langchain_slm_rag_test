import requests
import json
from typing import Dict, List, Optional
import datetime
import warnings
warnings.filterwarnings('ignore')

class GemmaAgent:
    def __init__(self):
        print("Gemma 에이전트를 초기화하는 중입니다...")
        
        self.base_url = "http://localhost:11434"
        self.model = "gemma3:1b"
        self.available_functions = {
            "get_current_weather": self.get_current_weather,
            "get_current_time": self.get_current_time,
            "calculate": self.calculate
        }
        
        print("에이전트 초기화가 완료되었습니다!")

    def get_current_weather(self, location: str) -> Dict:
        """현재 날씨 정보를 반환하는 가상 함수"""
        return {
            "location": location,
            "temperature": "20°C",
            "condition": "맑음",
            "humidity": "60%"
        }

    def get_current_time(self) -> str:
        """현재 시간을 반환"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate(self, expression: str) -> float:
        """간단한 수식 계산"""
        try:
            return eval(expression)
        except:
            return "계산할 수 없는 수식입니다."

    def extract_function_call(self, text: str) -> Optional[Dict]:
        """텍스트에서 함수 호출 의도를 추출"""
        try:
            # 함수 호출 의도 확인을 위한 프롬프트
            prompt = f"""
            다음 텍스트에서 함수 호출 의도를 파악하여 JSON 형식으로 반환하세요.
            사용 가능한 함수들:
            - get_current_weather(location: str)
            - get_current_time()
            - calculate(expression: str)

            텍스트: {text}

            JSON 형식으로 응답하세요:
            {{
                "function": "함수이름",
                "parameters": {{함수 파라미터}}
            }}
            """

            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "당신은 함수 호출을 파악하는 AI 어시스턴트입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                }
            )

            response_text = response.json()["message"]["content"]
            
            # JSON 부분 추출 시도
            try:
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            except:
                return None

        except Exception as e:
            print(f"함수 호출 의도 파악 중 오류: {str(e)}")
            return None

    def execute_function(self, function_info: Dict) -> str:
        """파악된 함수를 실행"""
        try:
            function_name = function_info.get("function")
            parameters = function_info.get("parameters", {})

            if function_name in self.available_functions:
                result = self.available_functions[function_name](**parameters)
                return str(result)
            else:
                return "지원하지 않는 함수입니다."

        except Exception as e:
            return f"함수 실행 중 오류 발생: {str(e)}"

    def chat(self, user_input: str) -> str:
        """사용자 입력 처리 및 응답 생성"""
        try:
            # 함수 호출 의도 파악
            function_info = self.extract_function_call(user_input)
            
            if function_info:
                # 함수 실행
                result = self.execute_function(function_info)
                
                # 결과를 자연스러운 응답으로 변환
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "함수 실행 결과를 자연스러운 한국어로 설명하세요."},
                            {"role": "user", "content": f"함수 실행 결과: {result}"}
                        ]
                    }
                )
                
                return response.json()["message"]["content"]
            else:
                # 일반 대화 응답
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 응답해주세요."},
                            {"role": "user", "content": user_input}
                        ]
                    }
                )
                
                return response.json()["message"]["content"]

        except Exception as e:
            return f"응답 생성 중 오류 발생: {str(e)}"

def main():
    print("Gemma 에이전트 시스템을 시작합니다...")
    
    try:
        agent = GemmaAgent()
        
        print("\n대화를 시작합니다! (종료하려면 'q' 또는 'quit'를 입력하세요)")
        print("사용 가능한 함수:")
        print("1. 날씨 확인: '서울의 날씨 알려줘'")
        print("2. 현재 시간: '지금 시간이 몇 시야?'")
        print("3. 계산: '23 + 45 계산해줘'")
        
        while True:
            user_input = input("\n사용자: ").strip()
            
            if user_input.lower() in ['q', 'quit']:
                print("\n대화를 종료합니다.")
                break
                
            if user_input:
                print("\n챗봇: ", end='')
                response = agent.chat(user_input)
                print(response)
            else:
                print("입력을 확인해주세요.")
                
    except Exception as e:
        print(f"\n시스템 오류 발생: {str(e)}")
        print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main() 