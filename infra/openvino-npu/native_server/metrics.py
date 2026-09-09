from __future__ import annotations

import threading
import time


class Metrics:
    BUCKETS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = {"embedding": 0, "rerank": 0}
        self.latencies_ms = {"embedding": [], "rerank": []}
        self.histograms = {
            metric: {kind: [0] * len(self.BUCKETS) for kind in self.requests}
            for metric in ("request_duration", "queue_wait", "inference_duration")
        }
        self.sums = {metric: {kind: 0.0 for kind in self.requests} for metric in self.histograms}
        self.counts = {metric: {kind: 0 for kind in self.requests} for metric in self.histograms}

    def observe(self, kind: str, request_seconds: float, queue_seconds: float, inference_seconds: float) -> None:
        with self._lock:
            self.requests[kind] += 1
            self.latencies_ms[kind].append(request_seconds * 1000)
            for metric, seconds in (("request_duration", request_seconds), ("queue_wait", queue_seconds), ("inference_duration", inference_seconds)):
                self.sums[metric][kind] += seconds
                self.counts[metric][kind] += 1
                for index, bucket in enumerate(self.BUCKETS):
                    if seconds <= bucket:
                        self.histograms[metric][kind][index] += 1

    def prometheus(self) -> str:
        with self._lock:
            lines = []
            for kind in self.requests:
                lines.append(f'native_npu_requests_total{{kind="{kind}"}} {self.requests[kind]}')
                lines.append(f'native_npu_latency_ms_last{{kind="{kind}"}} {self.latencies_ms[kind][-1] if self.latencies_ms[kind] else 0:.3f}')
                for metric in self.histograms:
                    for index, bucket in enumerate(self.BUCKETS):
                        lines.append(f'npu_{metric}_seconds_bucket{{operation="{kind}",le="{bucket}"}} {self.histograms[metric][kind][index]}')
                    lines.append(f'npu_{metric}_seconds_bucket{{operation="{kind}",le="+Inf"}} {self.counts[metric][kind]}')
                    lines.append(f'npu_{metric}_seconds_sum{{operation="{kind}"}} {self.sums[metric][kind]:.6f}')
                    lines.append(f'npu_{metric}_seconds_count{{operation="{kind}"}} {self.counts[metric][kind]}')
            return "\n".join(lines) + "\n"
