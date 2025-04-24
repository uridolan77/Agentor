# LLM Gateway Agent System Monitoring

This directory contains monitoring configurations for the LLM Gateway Agent System, including Grafana dashboards and Prometheus configurations.

## Dashboards

The `dashboards` directory contains Grafana dashboard configurations that can be imported into your Grafana instance:

1. **LLM Request Monitoring** (`llm_request_monitoring.json`)
   - Monitors LLM request rates, latency, error rates, and cache performance
   - Provides insights into context window usage and throttling

2. **LLM Cost Monitoring** (`llm_cost_monitoring.json`)
   - Tracks token usage and estimated costs by provider and model
   - Monitors quota usage and rate limits
   - Provides cost breakdowns and trends

3. **Agent Performance Monitoring** (`agent_performance_monitoring.json`)
   - Monitors agent execution rates, latency, and error rates
   - Tracks decision-making patterns and tool usage
   - Provides insights into reasoning steps and accuracy

4. **System Health Monitoring** (`system_health_monitoring.json`)
   - Monitors service and dependency health
   - Tracks circuit breaker states
   - Provides system resource usage metrics (CPU, memory, disk)

## How to Use

### Importing Dashboards into Grafana

1. Open your Grafana instance
2. Navigate to Dashboards > Import
3. Upload the JSON file or paste its contents
4. Select the appropriate data source (Prometheus)
5. Click "Import"

### Setting Up Prometheus

Ensure that Prometheus is configured to scrape metrics from the LLM Gateway Agent System. Add the following to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'llm_gateway'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']  # Adjust to your service's metrics endpoint
```

### Metrics Collection

The LLM Gateway Agent System exposes metrics at the `/metrics` endpoint. These metrics are collected by Prometheus and visualized in Grafana.

## Dashboard Features

### LLM Request Monitoring

- **Request Rate**: Tracks the number of requests per second by provider and model
- **Response Latency**: Monitors p50 and p95 latency for LLM requests
- **Error Rate**: Shows the percentage of failed requests
- **Cache Hit Ratio**: Displays the effectiveness of the caching system
- **Context Window Usage**: Monitors how much of the available context window is being used

### LLM Cost Monitoring

- **Token Usage**: Tracks token consumption over time by provider and model
- **Estimated Cost**: Calculates the cost of LLM usage in USD
- **Token Distribution**: Shows the distribution of tokens across different models
- **Cost Distribution**: Displays the cost breakdown by model
- **Quota Usage**: Monitors the usage of provider quotas
- **Rate Limit Tracking**: Tracks remaining rate limits

### Agent Performance Monitoring

- **Execution Rate**: Monitors the number of agent executions per second
- **Response Latency**: Tracks p50 and p95 latency for agent responses
- **Error Rate**: Shows the percentage of failed agent executions
- **Decision Patterns**: Displays the distribution of agent decisions
- **Tool Usage**: Tracks which tools are being used by agents
- **Reasoning Steps**: Monitors the complexity of agent reasoning
- **Accuracy**: Tracks the accuracy of agent responses
- **Memory Usage**: Monitors how agents are using different memory types

### System Health Monitoring

- **Service Health**: Displays the health status of all services
- **Dependency Health**: Monitors the health of external dependencies
- **Circuit Breaker State**: Shows the state of circuit breakers
- **Service Uptime**: Tracks how long services have been running
- **Resource Usage**: Monitors CPU, memory, and disk usage
- **Thread Count**: Tracks the number of threads in use
- **Open Files**: Monitors the number of open files

## Alerting

You can set up alerts in Grafana based on these dashboards. Some recommended alerts:

- High error rates (> 5%)
- Excessive latency (p95 > 10s)
- Low cache hit ratio (< 30%)
- High quota usage (> 80%)
- Circuit breakers in open state
- High resource usage (CPU > 80%, Memory > 80%)

## Extending the Dashboards

These dashboards can be extended with additional panels or modified to suit your specific needs. The JSON files are well-structured and can be edited directly or through the Grafana UI.

## Troubleshooting

If metrics are not appearing in the dashboards:

1. Check that the metrics endpoint is accessible
2. Verify that Prometheus is scraping the endpoint correctly
3. Ensure that the correct metrics are being exposed
4. Check that the Grafana data source is properly configured

For more detailed troubleshooting, check the Prometheus and Grafana logs.
