FROM python:3.11-slim

# 비루트 유저로 실행 (보안/권장)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# (선택) 런타임 영구 저장 경로 지정: Spaces에서만 /data가 붙음
ENV HF_HOME=/data/.huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 의존성 먼저 설치(빌드 캐시 최적화)
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 앱 소스 복사
COPY --chown=user . .

# FastAPI를 0.0.0.0:7860에 바인딩 (Spaces도 이 포트로 봄)
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--forwarded-allow-ips", "*", "--proxy-headers"]
