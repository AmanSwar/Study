# MODULE 39 — Inference Observability & Production Systems

## 1. Introduction & Scope

Production ML inference systems operate in environments where users and business stakes are high: latency SLOs are tight (P99 <100ms), availability must exceed 99.9%, and model quality must be maintained over time as data shifts. Unlike offline evaluation, observability in production must be comprehensive, low-overhead, and actionable—telemetry is worthless if it makes the system 20% slower.

This module covers the complete observability stack for inference: per-request metrics (latency, throughput, queue depth), hardware monitoring (GPU utilization, memory, power), anomaly detection (latency spikes, memory leaks, OOM prevention), A/B testing frameworks for inference, model versioning, and zero-downtime model updates. We emphasize high-cardinality metrics (request-level, not just aggregate) and automated action (circuit breakers, auto-scaling, fallback activation).

The business case is clear: inference observability prevents costly incidents (undetected model degradation costs 10x more than detected and fixed). Proper instrumentation reduces MTTR (mean time to repair) from hours to minutes, and enables confident deployment velocity.

We target three deployment scenarios: (1) enterprise AI services (100K+ req/sec, SLO P99 <100ms), (2) consumer applications (high volume, strict latency), and (3) research/experimentation (rapid iteration with data-driven decisions).

---

## 2. Telemetry & Per-Request Metrics

### 2.1 OpenTelemetry for Inference

OpenTelemetry is the industry standard for observable systems. Here's a production-grade implementation:

```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import time
import functools
from typing import Dict, Any, Callable
from dataclasses import dataclass
import numpy as np

# Initialize telemetry
def init_otel_tracing(service_name: str = "inference-server"):
    """Initialize OpenTelemetry tracing and metrics."""

    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )

    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )

    # Metrics
    prometheus_reader = PrometheusMetricReader()
    resource = Resource.create({SERVICE_NAME: service_name})
    meterProvider = MeterProvider(resource=resource, metric_readers=[prometheus_reader])
    metrics.set_meter_provider(meterProvider)

    # Auto-instrumentation
    FlaskInstrumentor().instrument()
    RequestsInstrumentor().instrument()

    return trace.get_tracer(__name__), meterProvider.get_meter(__name__)

# Per-request instrumentation
@dataclass
class RequestMetrics:
    """Per-request metrics."""
    request_id: str
    timestamp: float
    user_id: str
    model_name: str
    input_size: int
    batch_size: int
    latency_ms: float
    queue_wait_ms: float
    inference_latency_ms: float
    output_size: int
    status_code: int  # 200, 499 (timeout), 500, etc.
    error_message: str = None

class InferenceInstrumentor:
    """Instrument inference requests with detailed metrics."""

    def __init__(self, service_name: str = "inference-server"):
        self.tracer, self.meter = init_otel_tracing(service_name)

        # Create metrics
        self.request_latency = self.meter.create_histogram(
            name="inference.request.latency_ms",
            description="Request latency in milliseconds",
            unit="ms"
        )

        self.queue_depth = self.meter.create_observable_gauge(
            name="inference.queue.depth",
            description="Current queue depth",
            callbacks=[self._get_queue_depth]
        )

        self.request_counter = self.meter.create_counter(
            name="inference.requests.total",
            description="Total requests",
            unit="1"
        )

        self.error_counter = self.meter.create_counter(
            name="inference.errors.total",
            description="Total errors",
            unit="1"
        )

        self.batch_size_histogram = self.meter.create_histogram(
            name="inference.batch.size",
            description="Batch sizes",
            unit="1"
        )

        # For queue depth observable
        self.current_queue_depth = 0

    def _get_queue_depth(self, options):
        """Callback for observable queue depth."""
        return [self.current_queue_depth]

    def instrument_request(
        self,
        fn: Callable,
        model_name: str = "default"
    ) -> Callable:
        """Decorator for instrumenting inference requests."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            # Create span for this request
            with self.tracer.start_as_current_span(f"inference.{model_name}") as span:
                request_id = kwargs.get('request_id', 'unknown')
                user_id = kwargs.get('user_id', 'unknown')

                # Set span attributes
                span.set_attribute("request_id", request_id)
                span.set_attribute("user_id", user_id)
                span.set_attribute("model_name", model_name)

                # Measure queue wait
                queue_wait_start = time.time()
                span.add_event("Acquired from queue")

                # Measure inference
                inference_start = time.time()
                try:
                    result = fn(*args, **kwargs)
                    status_code = 200
                except Exception as e:
                    self.error_counter.add(1)
                    span.record_exception(e)
                    status_code = 500
                    result = None

                inference_latency_ms = (time.time() - inference_start) * 1000
                queue_wait_ms = (inference_start - queue_wait_start) * 1000
                total_latency_ms = (time.time() - queue_wait_start) * 1000

                # Record metrics
                self.request_latency.record(total_latency_ms)
                self.request_counter.add(1)
                span.set_attribute("latency_ms", total_latency_ms)

                # Metrics for batch sizes
                batch_size = kwargs.get('batch_size', 1)
                self.batch_size_histogram.record(batch_size)

                # Construct metrics object
                metrics_obj = RequestMetrics(
                    request_id=request_id,
                    timestamp=time.time(),
                    user_id=user_id,
                    model_name=model_name,
                    input_size=kwargs.get('input_size', 0),
                    batch_size=batch_size,
                    latency_ms=total_latency_ms,
                    queue_wait_ms=queue_wait_ms,
                    inference_latency_ms=inference_latency_ms,
                    output_size=len(str(result)) if result else 0,
                    status_code=status_code
                )

                return result, metrics_obj

        return wrapper

# Usage example
instrumentor = InferenceInstrumentor(service_name="llm-inference")

@instrumentor.instrument_request(model_name="llama-7b")
def run_inference(prompt: str, request_id: str, user_id: str, batch_size: int = 1, input_size: int = 0):
    """Example inference function."""
    time.sleep(0.1)  # Simulate inference
    return f"Response to: {prompt}"
```

### 2.2 Custom Metrics & Histograms

```python
class PercentileMetrics:
    """
    Track percentile metrics efficiently (P50, P95, P99).
    Uses a sliding window algorithm to avoid storing all values.
    """

    def __init__(self, window_size: int = 10000):
        self.values = []
        self.window_size = window_size
        self.sum = 0
        self.count = 0

    def record(self, value: float):
        """Record a value."""
        self.values.append(value)
        self.sum += value
        self.count += 1

        # Keep sliding window
        if len(self.values) > self.window_size:
            removed = self.values.pop(0)
            self.sum -= removed

    def percentile(self, p: float) -> float:
        """Get percentile (0-100)."""
        if not self.values:
            return 0.0

        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * (p / 100.0))
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def mean(self) -> float:
        """Get mean."""
        return self.sum / self.count if self.count > 0 else 0.0

    def stats(self) -> dict:
        """Return comprehensive statistics."""
        return {
            'count': self.count,
            'mean': self.mean(),
            'p50': self.percentile(50),
            'p95': self.percentile(95),
            'p99': self.percentile(99),
            'min': min(self.values) if self.values else 0,
            'max': max(self.values) if self.values else 0
        }

class MetricsCollector:
    """
    Collect and aggregate inference metrics.
    """

    def __init__(self, collection_window_s: int = 60):
        self.collection_window_s = collection_window_s
        self.latencies = PercentileMetrics()
        self.throughput_counter = 0
        self.error_counter = 0
        self.gpu_utilization = PercentileMetrics()
        self.memory_usage = PercentileMetrics()

        self.last_report_time = time.time()

    def record_request(self, latency_ms: float, success: bool = True):
        """Record a request."""
        self.latencies.record(latency_ms)
        self.throughput_counter += 1

        if not success:
            self.error_counter += 1

    def record_gpu_metric(self, utilization: float, memory_gb: float):
        """Record GPU metrics."""
        self.gpu_utilization.record(utilization)
        self.memory_usage.record(memory_gb)

    def get_report(self) -> dict:
        """Get metrics report."""
        now = time.time()
        elapsed = now - self.last_report_time

        return {
            'timestamp': now,
            'window_s': elapsed,
            'throughput_req_s': self.throughput_counter / elapsed if elapsed > 0 else 0,
            'error_rate': self.error_counter / self.throughput_counter if self.throughput_counter > 0 else 0,
            'latency': self.latencies.stats(),
            'gpu': {
                'utilization': self.gpu_utilization.stats(),
                'memory_gb': self.memory_usage.stats()
            }
        }

    def reset(self):
        """Reset for next window."""
        self.throughput_counter = 0
        self.error_counter = 0
        self.last_report_time = time.time()
```

---

## 3. Hardware Monitoring

### 3.1 GPU Telemetry with NVIDIA Management Library

```python
import subprocess
import json
import re
from typing import Dict, List
import time

class GPUMonitor:
    """
    Monitor GPU metrics: utilization, memory, power, temperature.
    """

    def __init__(self):
        self.gpu_count = self._get_gpu_count()

    def _get_gpu_count(self) -> int:
        """Get number of GPUs."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True,
                text=True
            )
            return len(result.stdout.strip().split('\n'))
        except FileNotFoundError:
            return 0

    def get_gpu_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Get per-GPU metrics.
        Returns: {gpu_id: {utilization, memory_used, memory_total, power_draw, temperature}}
        """
        if self.gpu_count == 0:
            return {}

        query_string = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"

        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=' + query_string,
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            metrics = {}
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = [p.strip() for p in line.split(',')]
                gpu_id = int(parts[0])

                metrics[gpu_id] = {
                    'utilization_percent': float(parts[1]),
                    'memory_used_mb': float(parts[2]),
                    'memory_total_mb': float(parts[3]),
                    'power_draw_w': float(parts[4]),
                    'temperature_c': float(parts[5])
                }

            return metrics

        except Exception as e:
            print(f"[GPUMonitor] Error: {e}")
            return {}

    def get_process_metrics(self, pid: int) -> Dict[str, float]:
        """Get GPU metrics for a specific process."""
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    'pmon',
                    '-c', '1',
                    '-p', str(pid)
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Parse output
            for line in result.stdout.split('\n'):
                if str(pid) in line:
                    parts = line.split()
                    return {
                        'gpu_id': int(parts[1]),
                        'gpu_memory_mb': int(parts[4]),
                        'sm_utilization': int(parts[5]),
                        'memory_utilization': int(parts[6])
                    }

            return {}

        except Exception as e:
            print(f"[GPUMonitor] Error: {e}")
            return {}

class CPUMonitor:
    """Monitor CPU metrics using perf/psutil."""

    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None

    def get_cpu_metrics(self) -> Dict[str, float]:
        """Get CPU metrics."""
        if not self.psutil:
            return {}

        return {
            'cpu_percent': self.psutil.cpu_percent(interval=0.1),
            'memory_percent': self.psutil.virtual_memory().percent,
            'memory_available_gb': self.psutil.virtual_memory().available / (1024**3)
        }

    def get_process_metrics(self, pid: int) -> Dict[str, float]:
        """Get process-specific metrics."""
        if not self.psutil:
            return {}

        try:
            p = self.psutil.Process(pid)
            with p.oneshot():
                return {
                    'cpu_percent': p.cpu_percent(),
                    'memory_mb': p.memory_info().rss / (1024**2),
                    'num_threads': p.num_threads(),
                    'io_read_mb': p.io_counters().read_bytes / (1024**2),
                    'io_write_mb': p.io_counters().write_bytes / (1024**2)
                }
        except:
            return {}

class HardwareMonitoringDaemon:
    """Background daemon for hardware monitoring."""

    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self.gpu_monitor = GPUMonitor()
        self.cpu_monitor = CPUMonitor()
        self.metrics_history = []
        self.running = False

    async def run(self):
        """Run monitoring loop."""
        self.running = True

        while self.running:
            metrics = {
                'timestamp': time.time(),
                'gpu': self.gpu_monitor.get_gpu_metrics(),
                'cpu': self.cpu_monitor.get_cpu_metrics()
            }

            self.metrics_history.append(metrics)

            # Keep last 1000 measurements
            if len(self.metrics_history) > 1000:
                self.metrics_history.pop(0)

            await asyncio.sleep(self.interval_s)

    def stop(self):
        """Stop monitoring."""
        self.running = False

    def get_current_metrics(self) -> Dict:
        """Get latest metrics."""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]

    def get_metrics_stats(self, window_s: float = 60) -> Dict:
        """Get aggregate statistics over time window."""
        now = time.time()
        cutoff = now - window_s

        recent = [m for m in self.metrics_history if m['timestamp'] >= cutoff]

        if not recent:
            return {}

        # Aggregate GPU metrics
        gpu_utils = []
        gpu_mems = []
        cpu_utils = []

        for m in recent:
            for gpu_id, gpu_m in m['gpu'].items():
                gpu_utils.append(gpu_m['utilization_percent'])
                gpu_mems.append(gpu_m['memory_used_mb'] / gpu_m['memory_total_mb'] * 100)

            if 'cpu_percent' in m['cpu']:
                cpu_utils.append(m['cpu']['cpu_percent'])

        return {
            'window_s': window_s,
            'gpu_utilization_mean': np.mean(gpu_utils) if gpu_utils else 0,
            'gpu_utilization_p95': np.percentile(gpu_utils, 95) if gpu_utils else 0,
            'gpu_memory_percent_mean': np.mean(gpu_mems) if gpu_mems else 0,
            'cpu_percent_mean': np.mean(cpu_utils) if cpu_utils else 0
        }
```

---

## 4. Anomaly Detection & Alerting

### 4.1 Latency Anomaly Detection

```python
import numpy as np
from scipy import stats

class LatencyAnomalyDetector:
    """
    Detect latency anomalies using statistical methods.
    """

    def __init__(
        self,
        window_size: int = 1000,
        z_score_threshold: float = 3.0,
        ewma_alpha: float = 0.3
    ):
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.ewma_alpha = ewma_alpha

        self.latencies = []
        self.baseline_mean = None
        self.baseline_std = None
        self.ewma_value = None

    def update(self, latency_ms: float) -> dict:
        """
        Update detector with new latency.
        Returns: {is_anomaly, anomaly_type, severity}
        """
        self.latencies.append(latency_ms)

        # Keep sliding window
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)

        # Initialize baseline after warming up
        if len(self.latencies) >= 100 and self.baseline_mean is None:
            self.baseline_mean = np.mean(self.latencies[:100])
            self.baseline_std = np.std(self.latencies[:100])
            self.ewma_value = self.baseline_mean

        if self.baseline_mean is None:
            return {'is_anomaly': False}

        # Update EWMA
        self.ewma_value = (
            self.ewma_alpha * latency_ms +
            (1 - self.ewma_alpha) * self.ewma_value
        )

        # Z-score based detection
        z_score = abs((latency_ms - self.baseline_mean) / (self.baseline_std + 1e-6))
        is_z_score_anomaly = z_score > self.z_score_threshold

        # Trend detection: is EWMA drifting?
        ewma_deviation = abs(self.ewma_value - self.baseline_mean) / (self.baseline_std + 1e-6)
        is_trend_anomaly = ewma_deviation > 2.0

        is_anomaly = is_z_score_anomaly or is_trend_anomaly

        severity = 'low'
        anomaly_type = None

        if is_z_score_anomaly:
            anomaly_type = 'spike'
            severity = 'high' if z_score > 5.0 else 'medium'

        if is_trend_anomaly:
            anomaly_type = 'drift'
            severity = 'high' if ewma_deviation > 3.0 else 'medium'

        return {
            'is_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'severity': severity,
            'z_score': z_score,
            'ewma_deviation': ewma_deviation,
            'baseline_mean': self.baseline_mean,
            'current_latency': latency_ms
        }

class OOMPredictor:
    """
    Predict out-of-memory (OOM) conditions before they occur.
    """

    def __init__(self, window_size: int = 100, threshold_fraction: float = 0.85):
        self.window_size = window_size
        self.threshold_fraction = threshold_fraction
        self.memory_observations = []
        self.gpu_total_memory_gb = None

    def set_total_memory(self, total_gb: float):
        """Set GPU total memory."""
        self.gpu_total_memory_gb = total_gb

    def update(self, memory_used_gb: float, memory_total_gb: float = None) -> dict:
        """Update with memory observation."""
        if memory_total_gb:
            self.gpu_total_memory_gb = memory_total_gb

        self.memory_observations.append(memory_used_gb)

        if len(self.memory_observations) > self.window_size:
            self.memory_observations.pop(0)

        current_fraction = memory_used_gb / (self.gpu_total_memory_gb or 1.0)

        # Fit linear trend
        if len(self.memory_observations) >= 10:
            x = np.arange(len(self.memory_observations))
            y = np.array(self.memory_observations)

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Predict when OOM would occur
            oom_threshold = (self.gpu_total_memory_gb or 40) * self.threshold_fraction
            if slope > 0:
                iterations_to_oom = (oom_threshold - memory_used_gb) / slope
            else:
                iterations_to_oom = float('inf')

            return {
                'memory_used_gb': memory_used_gb,
                'memory_fraction': current_fraction,
                'trend_slope_per_iter': slope,
                'r_squared': r_value ** 2,
                'iterations_to_oom': iterations_to_oom,
                'oom_imminent': current_fraction > self.threshold_fraction,
                'oom_predicted_in_iterations': int(iterations_to_oom) if iterations_to_oom < float('inf') else None
            }

        return {
            'memory_used_gb': memory_used_gb,
            'memory_fraction': current_fraction,
            'oom_imminent': current_fraction > self.threshold_fraction
        }

class AlertingSystem:
    """Route anomalies to alerts."""

    def __init__(self):
        self.alerts = []

    def check_latency_anomaly(self, anomaly_info: dict):
        """Check latency anomaly and issue alert if severe."""
        if anomaly_info.get('severity') == 'high':
            alert = {
                'type': 'LATENCY_ANOMALY',
                'severity': 'HIGH',
                'message': f"Latency spike detected: {anomaly_info['current_latency']:.0f}ms "
                          f"(baseline: {anomaly_info['baseline_mean']:.0f}ms)",
                'timestamp': time.time(),
                'details': anomaly_info
            }
            self.alerts.append(alert)
            print(f"[ALERT] {alert['message']}")

    def check_oom_prediction(self, oom_info: dict):
        """Check OOM prediction and issue alert."""
        if oom_info.get('oom_imminent'):
            alert = {
                'type': 'OOM_WARNING',
                'severity': 'CRITICAL',
                'message': f"OOM predicted in {oom_info.get('iterations_to_oom', '?')} iterations",
                'timestamp': time.time(),
                'details': oom_info
            }
            self.alerts.append(alert)
            print(f"[ALERT] {alert['message']}")
            # TODO: Trigger mitigation (reduce batch size, etc.)

    def get_recent_alerts(self, window_s: float = 300) -> List[dict]:
        """Get recent alerts within time window."""
        cutoff = time.time() - window_s
        return [a for a in self.alerts if a['timestamp'] >= cutoff]
```

---

## 5. A/B Testing for Inference

### 5.1 Multi-Armed Bandit Routing

```python
import random
from enum import Enum

class ABTestStrategy(Enum):
    FIXED_SPLIT = "fixed_split"  # Static 50/50
    EPSILON_GREEDY = "epsilon_greedy"  # Explore/exploit
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian

class ABTestRouter:
    """
    Route requests between model A and B for A/B testing.
    Supports multiple strategies.
    """

    def __init__(
        self,
        strategy: ABTestStrategy = ABTestStrategy.FIXED_SPLIT,
        model_a_weight: float = 0.5,
        epsilon: float = 0.1
    ):
        self.strategy = strategy
        self.model_a_weight = model_a_weight
        self.epsilon = epsilon

        # Metrics
        self.model_a_latencies = []
        self.model_b_latencies = []
        self.model_a_errors = 0
        self.model_b_errors = 0
        self.model_a_success = 0
        self.model_b_success = 0

    def choose_model(self) -> str:
        """Choose which model to use."""
        if self.strategy == ABTestStrategy.FIXED_SPLIT:
            return "model_a" if random.random() < self.model_a_weight else "model_b"

        elif self.strategy == ABTestStrategy.EPSILON_GREEDY:
            if random.random() < self.epsilon:
                # Explore
                return random.choice(["model_a", "model_b"])
            else:
                # Exploit: choose model with lower latency
                a_latency = np.mean(self.model_a_latencies[-100:]) if self.model_a_latencies else float('inf')
                b_latency = np.mean(self.model_b_latencies[-100:]) if self.model_b_latencies else float('inf')
                return "model_a" if a_latency < b_latency else "model_b"

        elif self.strategy == ABTestStrategy.THOMPSON_SAMPLING:
            # Bayesian approach: sample from posterior
            a_success_rate = self.model_a_success / max(1, self.model_a_success + self.model_a_errors)
            b_success_rate = self.model_b_success / max(1, self.model_b_success + self.model_b_errors)

            # Beta distribution sampling
            a_sample = np.random.beta(self.model_a_success + 1, self.model_a_errors + 1)
            b_sample = np.random.beta(self.model_b_success + 1, self.model_b_errors + 1)

            return "model_a" if a_sample > b_sample else "model_b"

    def record_result(
        self,
        model: str,
        latency_ms: float,
        success: bool = True
    ):
        """Record inference result."""
        if model == "model_a":
            self.model_a_latencies.append(latency_ms)
            if success:
                self.model_a_success += 1
            else:
                self.model_a_errors += 1

        else:
            self.model_b_latencies.append(latency_ms)
            if success:
                self.model_b_success += 1
            else:
                self.model_b_errors += 1

    def get_test_results(self) -> dict:
        """Get A/B test results."""
        a_count = self.model_a_success + self.model_a_errors
        b_count = self.model_b_success + self.model_b_errors

        a_success_rate = self.model_a_success / a_count if a_count > 0 else 0
        b_success_rate = self.model_b_success / b_count if b_count > 0 else 0

        a_latency = np.mean(self.model_a_latencies) if self.model_a_latencies else 0
        b_latency = np.mean(self.model_b_latencies) if self.model_b_latencies else 0

        return {
            'model_a': {
                'count': a_count,
                'success_rate': a_success_rate,
                'latency_mean_ms': a_latency,
                'error_rate': 1 - a_success_rate
            },
            'model_b': {
                'count': b_count,
                'success_rate': b_success_rate,
                'latency_mean_ms': b_latency,
                'error_rate': 1 - b_success_rate
            },
            'winner': 'model_a' if a_latency < b_latency else 'model_b',
            'latency_improvement': abs(a_latency - b_latency) / max(a_latency, b_latency) * 100
        }

class StatisticalSignificanceTest:
    """Test if A/B test result is statistically significant."""

    @staticmethod
    def t_test(
        latencies_a: List[float],
        latencies_b: List[float],
        confidence: float = 0.95
    ) -> dict:
        """Perform t-test on latencies."""
        t_stat, p_value = stats.ttest_ind(latencies_a, latencies_b)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < (1 - confidence),
            'confidence': confidence
        }
```

---

## 6. Model Versioning & Hot Swap

### 6.1 Zero-Downtime Model Replacement

```python
from enum import Enum
import threading

class ModelVersion:
    """Represents a model version."""

    def __init__(self, version_id: str, model_path: str, metadata: dict = None):
        self.version_id = version_id
        self.model_path = model_path
        self.metadata = metadata or {}
        self.loaded_model = None
        self.creation_time = time.time()
        self.request_count = 0
        self.error_count = 0

class ModelRegistry:
    """
    Manage multiple model versions with atomic swaps.
    """

    def __init__(self):
        self.versions: Dict[str, ModelVersion] = {}
        self.active_version_id = None
        self.lock = threading.RLock()

    def register_version(
        self,
        version_id: str,
        model_path: str,
        metadata: dict = None,
        load_fn=None
    ) -> bool:
        """Register a new model version."""
        with self.lock:
            if version_id in self.versions:
                print(f"[ModelRegistry] Version {version_id} already exists")
                return False

            version = ModelVersion(version_id, model_path, metadata)

            # Load model
            if load_fn:
                try:
                    version.loaded_model = load_fn(model_path)
                    print(f"[ModelRegistry] Loaded version {version_id}")
                except Exception as e:
                    print(f"[ModelRegistry] Failed to load {version_id}: {e}")
                    return False

            self.versions[version_id] = version

            # Set as active if first version
            if self.active_version_id is None:
                self.active_version_id = version_id
                print(f"[ModelRegistry] Set {version_id} as active")

            return True

    def set_active_version(self, version_id: str) -> bool:
        """
        Atomically swap to new active version.
        No requests are dropped.
        """
        with self.lock:
            if version_id not in self.versions:
                print(f"[ModelRegistry] Version {version_id} not found")
                return False

            old_version_id = self.active_version_id
            self.active_version_id = version_id

            print(f"[ModelRegistry] Swapped from {old_version_id} to {version_id}")
            return True

    def get_active_model(self):
        """Get currently active model."""
        with self.lock:
            if self.active_version_id:
                return self.versions[self.active_version_id].loaded_model
            return None

    def record_request(self, version_id: str, success: bool = True):
        """Record request for version."""
        with self.lock:
            if version_id in self.versions:
                self.versions[version_id].request_count += 1
                if not success:
                    self.versions[version_id].error_count += 1

    def get_version_stats(self) -> dict:
        """Get statistics for all versions."""
        with self.lock:
            stats = {}
            for vid, version in self.versions.items():
                error_rate = (
                    version.error_count / version.request_count
                    if version.request_count > 0 else 0
                )
                stats[vid] = {
                    'request_count': version.request_count,
                    'error_count': version.error_count,
                    'error_rate': error_rate,
                    'is_active': vid == self.active_version_id,
                    'created_s_ago': time.time() - version.creation_time
                }
            return stats

class CanaryRollout:
    """
    Gradual model rollout: route small % of traffic to new version first.
    """

    def __init__(self, router: ABTestRouter):
        self.router = router
        self.canary_percentage = 0

    def start_canary(
        self,
        new_version_id: str,
        initial_percentage: float = 5.0,
        increment_percentage: float = 10.0,
        increment_interval_s: float = 300
    ):
        """
        Start canary rollout.
        Gradually increase traffic percentage over time.
        """
        self.canary_percentage = initial_percentage
        self.new_version_id = new_version_id
        self.increment_percentage = increment_percentage
        self.increment_interval_s = increment_interval_s
        self.canary_start_time = time.time()

        print(f"[Canary] Starting rollout of {new_version_id} at {initial_percentage}%")

    def should_promote_to_stable(self, error_threshold: float = 0.01) -> bool:
        """
        Check if canary should be promoted to stable.
        Returns True if error rate is low and canary is at 100%.
        """
        elapsed = time.time() - self.canary_start_time

        # Auto-increment canary percentage
        auto_percentage = (
            self.canary_percentage +
            (elapsed // self.increment_interval_s) * self.increment_percentage
        )
        self.canary_percentage = min(100, auto_percentage)

        # Check error rate
        stats = self.router.get_test_results()
        if stats['model_b']['error_rate'] > error_threshold:
            print(f"[Canary] Error rate too high, halting rollout")
            return False

        if self.canary_percentage >= 100:
            print(f"[Canary] Reached 100%, ready for promotion")
            return True

        return False
```

---

## 7. Production Incident Response

### 7.1 Circuit Breaker & Fallback

```python
from enum import Enum

class CircuitBreakerState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Stop forwarding requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """
    Circuit breaker pattern for inference.
    Prevents cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int = 10,
        recovery_timeout_s: float = 60.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()

    def call(self, fn, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout elapsed
                if (time.time() - self.last_failure_time) > self.recovery_timeout_s:
                    self.state = CircuitBreakerState.HALF_OPEN
                    print(f"[CircuitBreaker] State -> HALF_OPEN (testing recovery)")
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = fn(*args, **kwargs)

            with self.lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    print(f"[CircuitBreaker] State -> CLOSED (recovered)")

            return result

        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    print(f"[CircuitBreaker] State -> OPEN (failure threshold reached)")

            raise

class FallbackPolicy:
    """Fallback strategies when primary model fails."""

    STRATEGIES = {
        'cache': 'Return cached response from last successful inference',
        'fallback_model': 'Use lighter/older model version',
        'degraded_mode': 'Return partial result with reduced quality',
        'error': 'Return error to user'
    }

    def __init__(self, strategy: str = 'fallback_model'):
        self.strategy = strategy
        self.response_cache = {}

    def execute(
        self,
        request_id: str,
        cache_hit=None,
        fallback_fn=None
    ):
        """Execute fallback strategy."""
        if self.strategy == 'cache' and cache_hit:
            return cache_hit

        elif self.strategy == 'fallback_model' and fallback_fn:
            return fallback_fn()

        elif self.strategy == 'degraded_mode':
            return {'response': 'Service degraded, partial results only'}

        else:
            raise Exception("No fallback available")
```

---

## 8. Monitoring Dashboard & Reporting

```python
class MonitoringDashboard:
    """
    Summary dashboard for production inference systems.
    """

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def print_status(self):
        """Print current system status."""
        report = self.collector.get_report()

        print("\n" + "="*60)
        print("INFERENCE MONITORING DASHBOARD")
        print("="*60)

        print(f"\nTimestamp: {time.ctime(report['timestamp'])}")

        print("\n[THROUGHPUT & ERRORS]")
        print(f"  Throughput: {report['throughput_req_s']:.0f} req/s")
        print(f"  Error rate: {report['error_rate']*100:.2f}%")

        print("\n[LATENCY PERCENTILES]")
        lat = report['latency']
        print(f"  P50: {lat['p50']:.1f}ms")
        print(f"  P95: {lat['p95']:.1f}ms")
        print(f"  P99: {lat['p99']:.1f}ms")
        print(f"  Mean: {lat['mean']:.1f}ms")

        print("\n[GPU METRICS]")
        gpu = report['gpu']
        print(f"  Utilization: {gpu['utilization']['mean']:.1f}% "
              f"(P95: {gpu['utilization']['p95']:.1f}%)")
        print(f"  Memory: {gpu['memory_gb']['mean']:.1f}GB "
              f"(P95: {gpu['memory_gb']['p95']:.1f}GB)")

        print("="*60 + "\n")

    def write_metrics_to_file(self, filepath: str):
        """Write detailed metrics to file for analysis."""
        report = self.collector.get_report()

        with open(filepath, 'a') as f:
            f.write(f"{report['timestamp']},{report['throughput_req_s']:.0f},"
                   f"{report['latency']['p99']:.1f},{report['error_rate']:.4f}\n")
```

---

## 9. Production Checklist

```python
class ProductionChecklistObservability:
    """Production readiness for observability."""

    checklist = {
        'Telemetry': [
            'OpenTelemetry configured for request tracing',
            'Per-request metrics logged (latency, throughput, errors)',
            'Batch size distribution tracked',
            'Model version in every trace'
        ],
        'Hardware Monitoring': [
            'GPU utilization tracked per GPU',
            'Memory usage monitored (GPU + CPU)',
            'Temperature alerts configured',
            'Power consumption tracked for cost analysis'
        ],
        'Anomaly Detection': [
            'Latency spike detection (Z-score based)',
            'Memory leak detection (trend analysis)',
            'OOM predictor enabled with early alerts',
            'Error rate anomalies tracked'
        ],
        'A/B Testing': [
            'A/B test framework deployed',
            'Statistical significance testing in place',
            'Proper routing (no request duplication)',
            'Results published daily'
        ],
        'Model Management': [
            'Model versioning in place',
            'Canary rollout procedure documented',
            'Zero-downtime swap tested',
            'Rollback procedure tested'
        ],
        'Incident Response': [
            'Circuit breaker configured',
            'Fallback model available',
            'On-call playbooks written',
            'Alert thresholds tuned'
        ],
        'Dashboards': [
            'Real-time latency dashboard',
            'Error rate dashboard with drill-down',
            'Model comparison dashboard',
            'Hardware utilization dashboard'
        ]
    }

    @staticmethod
    def print_checklist():
        for category, items in ProductionChecklistObservability.checklist.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  [ ] {item}")
```

---

## 10. Summary & Key Takeaways

This module covered comprehensive observability for production inference:

1. **Telemetry**: OpenTelemetry for distributed tracing, per-request metrics
2. **Hardware monitoring**: GPU/CPU utilization, memory, power, temperature
3. **Anomaly detection**: Latency spikes, drift, memory leaks, OOM prediction
4. **A/B testing**: Multi-armed bandit strategies, statistical significance
5. **Model versioning**: Atomic swaps, canary rollouts, zero-downtime updates
6. **Incident response**: Circuit breakers, fallback strategies, graceful degradation
7. **Dashboards**: Real-time visibility into system health and performance

**Key metrics to track:**
- Latency: P50/P95/P99 (target <100ms for interactive)
- Throughput: requests/sec
- Error rate: <0.1% for production
- GPU utilization: 50-80% optimal (higher = wasted compute, lower = underutilized)
- Model version: track requests per version for safe rollouts

**Next steps**: Deploy OpenTelemetry, set up hardware monitoring daemon, implement latency anomaly detection, and configure A/B testing framework. Measure baseline metrics before tuning SLOs and alert thresholds.
