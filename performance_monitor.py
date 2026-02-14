"""Performance monitoring and metrics tracking."""

import time
import psutil
from typing import List, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    processing_time: float
    cpu_percent: float
    memory_percent: float
    timestamp: float


class PerformanceMonitor:
    """Monitor CPU, memory, and frame processing performance."""

    def __init__(self, max_samples: int = 100):
        """Initialize performance monitor.
        
        Args:
            max_samples: Maximum number of samples to keep
        """
        self.max_samples = max_samples
        self.frame_times: deque = deque(maxlen=max_samples)
        self.cpu_samples: deque = deque(maxlen=max_samples)
        self.memory_samples: deque = deque(maxlen=max_samples)
        self.start_time = time.time()
        self.last_metrics_time = time.time()
        self.frame_count = 0
        self.total_processing_time = 0.0

    def record_frame(self, processing_time: float):
        """Record metrics for a processed frame.
        
        Args:
            processing_time: Time taken to process frame in seconds
        """
        self.frame_count += 1
        self.total_processing_time += processing_time
        self.frame_times.append(processing_time)
        
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        
        self.cpu_samples.append(cpu_percent)
        self.memory_samples.append(memory_percent)

    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage.
        
        Returns:
            CPU usage percentage
        """
        return psutil.cpu_percent(interval=None)

    def get_memory_percent(self) -> float:
        """Get current memory usage percentage.
        
        Returns:
            Memory usage percentage
        """
        return psutil.virtual_memory().percent

    def get_avg_processing_time(self) -> float:
        """Get average frame processing time.
        
        Returns:
            Average processing time in seconds
        """
        if not self.frame_times:
            return 0.0
        return sum(self.frame_times) / len(self.frame_times)

    def get_min_processing_time(self) -> float:
        """Get minimum frame processing time.
        
        Returns:
            Minimum processing time in seconds
        """
        if not self.frame_times:
            return 0.0
        return min(self.frame_times)

    def get_max_processing_time(self) -> float:
        """Get maximum frame processing time.
        
        Returns:
            Maximum processing time in seconds
        """
        if not self.frame_times:
            return 0.0
        return max(self.frame_times)

    def get_fps(self) -> float:
        """Get effective frames per second.
        
        Returns:
            Frames per second
        """
        if self.frame_count == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        
        return self.frame_count / elapsed

    def get_avg_cpu(self) -> float:
        """Get average CPU usage.
        
        Returns:
            Average CPU usage percentage
        """
        if not self.cpu_samples:
            return 0.0
        return sum(self.cpu_samples) / len(self.cpu_samples)

    def get_avg_memory(self) -> float:
        """Get average memory usage.
        
        Returns:
            Average memory usage percentage
        """
        if not self.memory_samples:
            return 0.0
        return sum(self.memory_samples) / len(self.memory_samples)

    def get_metrics_summary(self) -> str:
        """Get formatted metrics summary.
        
        Returns:
            Formatted string with metrics
        """
        avg_time = self.get_avg_processing_time()
        min_time = self.get_min_processing_time()
        max_time = self.get_max_processing_time()
        fps = self.get_fps()
        avg_cpu = self.get_avg_cpu()
        avg_memory = self.get_avg_memory()
        current_cpu = self.get_cpu_percent()
        current_memory = self.get_memory_percent()
        
        return (
            f"Performance Metrics:\n"
            f"  FPS: {fps:.2f}\n"
            f"  Frame Time: avg={avg_time*1000:.1f}ms, min={min_time*1000:.1f}ms, max={max_time*1000:.1f}ms\n"
            f"  CPU: current={current_cpu:.1f}%, avg={avg_cpu:.1f}%\n"
            f"  Memory: current={current_memory:.1f}%, avg={avg_memory:.1f}%"
        )

    def reset(self):
        """Reset all metrics."""
        self.frame_times.clear()
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.start_time = time.time()
        self.last_metrics_time = time.time()
        self.frame_count = 0
        self.total_processing_time = 0.0

