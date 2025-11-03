# CortexRuntime Production Deployment Runbook

## 1. Bare Metal / Local Service

### Prerequisites
- Python >= 3.10
- PortAudio development libraries
- Local inference server (llama.cpp server or Ollama)

```bash
pip install -e .
python -m cortex start --config cortex.yaml
```

## 2. Docker Compose Deployment

```bash
docker-compose up -d
docker-compose logs -f cortex-engine
```

## 3. Kubernetes Helm Chart Deployment

```bash
helm upgrade --install cortex-runtime ./deploy/helm \
  --namespace cortex \
  --create-namespace \
  --set replicaCount=3 \
  --set autoscaling.enabled=true
```
