# app/api/services/metrics.py
from typing import Dict, Any
import time
import psutil
import torch

class MetricsService:
    """
    Service to track and expose server health and performance metrics.
    Minimal implementation for initial setup.
    """
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.total_latency = 0.0
        self.cuda_available = torch.cuda.is_available()

    def update_latency(self, latency: float):
        """Records the time taken for a successful inference request."""
        self.request_count += 1
        self.total_latency += latency

    def get_health_metrics(self) -> Dict[str, Any]:
        """Collects and returns health and performance metrics (/health endpoint)."""
        
        # Calculate derived metrics
        uptime = time.time() - self.start_time
        avg_latency_s = self.total_latency / self.request_count if self.request_count > 0 else 0
        throughput = self.request_count / uptime if uptime > 0 else 0

        # Core Metrics (as requested in the task [cite: 1])
        metrics = {
            "status": "ok",
            "uptime_seconds": round(uptime, 2),
            "request_count": self.request_count,
            "avg_latency_ms": round(avg_latency_s * 1000, 2),
            "throughput_req_per_s": round(throughput, 2),
            "cpu_util_percent": psutil.cpu_percent(),
        }

        # GPU Metrics (if available, as requested in the task [cite: 1])
        if self.cuda_available:
            try:
                # Use torch for simple memory stats (GB - Metric System)
                gpu_index = 0
                metrics["gpu_memory_used_gb"] = round(torch.cuda.memory_allocated(gpu_index) / (1024**3), 2)
                metrics["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3), 2)
                metrics["gpu_status"] = "active"
            except Exception:
                metrics["gpu_status"] = "error"
                
        return metrics