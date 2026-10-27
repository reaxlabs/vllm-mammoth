# Ready image: stock vLLM 0.32.0 + vllm-mammoth. Serves baselabs/mammoth-1 on /v1/systemone.
#   docker build -t vllm-mammoth:0.1.0 -f docker/Dockerfile .
#   docker run --gpus all --ipc host -p 8000:8000 -v /models/mammoth-1:/models/mammoth-1:ro vllm-mammoth:0.1.0
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
COPY . /opt/vllm-mammoth
RUN python3 -m pip install --no-deps --no-cache-dir /opt/vllm-mammoth && python3 -c "import vllm_mammoth"
ENV VLLM_BATCH_INVARIANT=1 \
    MAMMOTH_MODEL=/models/mammoth-1
COPY docker/serve.sh /usr/local/bin/serve-mammoth
ENTRYPOINT ["bash", "/usr/local/bin/serve-mammoth"]
