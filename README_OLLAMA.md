# Docker Compose를 사용한 Ollama Gemma 모델 실행 가이드

이 가이드는 Docker Compose를 사용하여 Ollama와 Gemma 모델을 실행하고 테스트하는 방법을 설명합니다.

## 1. 사전 준비

### 필수 설치 항목
- Docker
- Docker Compose
- curl (테스트용)

### 버전 확인
```bash
docker --version
docker compose version
```

## 2. 프로젝트 구조 설정

```bash
mkdir ollama-project
cd ollama-project
```

### docker-compose.yml 파일 생성

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:0.6.0
    container_name: ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G

volumes:
  ollama_data:
```

## 3. Ollama 서비스 실행

```bash
# 서비스 시작
docker compose up -d

# 로그 확인
docker compose logs -f
```

## 4. Gemma 모델 다운로드

```bash
# Docker 컨테이너에서 모델 다운로드
docker compose exec ollama ollama pull gemma3:1b
```

## 5. API 테스트

### 기본 텍스트 생성
```bash
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "gemma3:1b",
  "prompt": "한국의 수도는 어디인가요?"
}'
```

### 스트리밍 응답
```bash
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "gemma3:1b",
  "prompt": "한국의 수도는 어디인가요?",
  "stream": true
}'
```

### 채팅 API 사용
```bash
curl -X POST http://localhost:11434/api/chat -d '{
  "model": "gemma3:1b",
  "messages": [
    {
      "role": "user",
      "content": "한국의 수도는 어디인가요?"
    }
  ]
}'
```

## 6. Docker Compose 명령어

### 서비스 상태 확인
```bash
docker compose ps
```

### 리소스 사용량 모니터링
```bash
docker compose top
```

### 서비스 재시작
```bash
docker compose restart
```

### 서비스 중지
```bash
docker compose stop
```

### 서비스 및 볼륨 완전 제거
```bash
docker compose down -v
```

## 7. 문제 해결

### 로그 확인
```bash
docker compose logs ollama
```

### 모델 상태 확인
```bash
docker compose exec ollama ollama list
```

### 컨테이너 쉘 접속
```bash
docker compose exec ollama /bin/bash
```

## 8. 성능 최적화

### docker-compose.yml 리소스 설정 예시
```yaml
services:
  ollama:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## 9. API 응답 예시

```json
{
  "model": "gemma3:1b",
  "created_at": "2024-02-25T12:34:56.789Z",
  "response": "서울은 대한민국의 수도이자 최대 도시입니다.",
  "done": true
}
```

## 10. 환경 변수 설정 (선택사항)

### .env 파일 생성
```env
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*
NVIDIA_VISIBLE_DEVICES=all  # GPU 사용 시
```

### docker-compose.yml에 환경 변수 추가
```yaml
services:
  ollama:
    env_file:
      - .env
```

## 시스템 요구사항

- Docker Engine 20.10.0 이상
- Docker Compose 2.0.0 이상
- 최소 8GB RAM
- 20GB 이상의 저장 공간
- 인터넷 연결

## 주의사항

1. 모델 첫 다운로드 시 시간이 소요될 수 있습니다.
2. 메모리 설정은 시스템 사양에 맞게 조정하세요.
3. GPU 사용 시 추가 설정이 필요할 수 있습니다.
4. 볼륨 데이터는 영구 저장되므로 필요시 관리가 필요합니다.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
