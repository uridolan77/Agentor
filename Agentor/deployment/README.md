# Agentor Deployment

This directory contains deployment configurations for the Agentor framework.

## Docker Deployment

To deploy Agentor using Docker Compose:

1. Set up environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

2. Build and start the containers:

```bash
cd deployment
docker-compose up -d
```

3. Access the services:
   - Agentor API: http://localhost:8000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

## Kubernetes Deployment

To deploy Agentor on Kubernetes:

1. Create a namespace:

```bash
kubectl create namespace agentor
```

2. Create secrets:

```bash
kubectl create secret generic agentor-secrets \
  --namespace agentor \
  --from-literal=openai-api-key=your_openai_api_key \
  --from-literal=anthropic-api-key=your_anthropic_api_key
```

3. Apply the Kubernetes configurations:

```bash
kubectl apply -f kubernetes/configmap.yaml -n agentor
kubectl apply -f kubernetes/deployment.yaml -n agentor
kubectl apply -f kubernetes/service.yaml -n agentor
```

4. Check the deployment status:

```bash
kubectl get all -n agentor
```

## Monitoring

The deployment includes Prometheus and Grafana for monitoring:

- Prometheus collects metrics from the Agentor service
- Grafana provides dashboards for visualizing the metrics

The Grafana dashboard includes:
- Request rates and latencies
- Error rates
- Circuit breaker state changes
- Retry attempts
- Bulkhead rejections
- Timeout values
- Memory operations

## Configuration

The configuration is managed through:
- Environment variables
- ConfigMap in Kubernetes
- config.json file in Docker

Key configuration parameters:
- Server settings (host, port, workers)
- Logging settings
- Metrics settings
- Circuit breaker settings
- Retry settings
- Timeout settings
- Bulkhead settings
- Cache settings

## Scaling

The Agentor service can be scaled horizontally:

- In Docker Compose:
  ```bash
  docker-compose up -d --scale agentor=3
  ```

- In Kubernetes:
  ```bash
  kubectl scale deployment agentor -n agentor --replicas=3
  ```
