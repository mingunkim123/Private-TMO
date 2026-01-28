"""
Performance Profiler for Jetson Deployment

Measures:
1. Inference latency (local vs cloud)
2. GPU/CPU memory usage
3. Power consumption (Jetson-specific)
4. LoRA adapter switching overhead
5. End-to-end pipeline latency
"""

import time
import os
import sys
import json
import psutil
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager


@dataclass
class LatencyMeasurement:
    """Single latency measurement"""
    component: str
    duration_ms: float
    timestamp: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    gpu_used_mb: Optional[float] = None
    gpu_total_mb: Optional[float] = None


@dataclass
class PowerMeasurement:
    """Power consumption measurement (Jetson-specific)"""
    timestamp: float
    total_power_mw: float
    gpu_power_mw: Optional[float] = None
    cpu_power_mw: Optional[float] = None
    soc_power_mw: Optional[float] = None


@dataclass
class ProfileResult:
    """Complete profiling result"""
    name: str
    latencies: List[LatencyMeasurement]
    memory_snapshots: List[MemorySnapshot]
    power_measurements: List[PowerMeasurement]
    
    # Computed metrics
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_memory_mb: float = 0.0
    avg_power_mw: float = 0.0
    
    def compute_metrics(self):
        """Compute aggregate metrics"""
        if self.latencies:
            durations = [l.duration_ms for l in self.latencies]
            self.mean_latency_ms = sum(durations) / len(durations)
            
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            self.p50_latency_ms = sorted_durations[int(n * 0.5)]
            self.p95_latency_ms = sorted_durations[int(n * 0.95)]
            self.p99_latency_ms = sorted_durations[min(int(n * 0.99), n - 1)]
        
        if self.memory_snapshots:
            self.max_memory_mb = max(s.ram_used_mb for s in self.memory_snapshots)
        
        if self.power_measurements:
            self.avg_power_mw = sum(p.total_power_mw for p in self.power_measurements) / len(self.power_measurements)


class JetsonPowerMonitor:
    """
    Monitor power consumption on Jetson devices
    
    Reads from /sys/bus/i2c/drivers/ina3221x/ on Jetson
    """
    
    POWER_PATHS = {
        'orin': {
            'total': '/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in*_input',
            'gpu': '/sys/devices/gpu.0/railgate_enable',
        },
        'nano': {
            'total': '/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input',
            'gpu': '/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input',
            'cpu': '/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power2_input',
        },
        'tx2': {
            'total': '/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input',
            'gpu': '/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power1_input',
        }
    }
    
    def __init__(self, device_type: str = 'auto'):
        self.device_type = self._detect_device() if device_type == 'auto' else device_type
        self.available = self._check_availability()
    
    def _detect_device(self) -> str:
        """Detect Jetson device type"""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read().lower()
                if 'orin' in content:
                    return 'orin'
                elif 'nano' in content:
                    return 'nano'
                elif 'tx2' in content:
                    return 'tx2'
        except:
            pass
        return 'unknown'
    
    def _check_availability(self) -> bool:
        """Check if power monitoring is available"""
        if self.device_type not in self.POWER_PATHS:
            return False
        
        paths = self.POWER_PATHS[self.device_type]
        return any(os.path.exists(p.split('*')[0]) for p in paths.values())
    
    def read_power(self) -> PowerMeasurement:
        """Read current power consumption"""
        if not self.available:
            return PowerMeasurement(
                timestamp=time.time(),
                total_power_mw=0.0,
            )
        
        def read_file(path: str) -> Optional[float]:
            try:
                import glob
                matches = glob.glob(path)
                if matches:
                    with open(matches[0], 'r') as f:
                        return float(f.read().strip())
            except:
                pass
            return None
        
        paths = self.POWER_PATHS.get(self.device_type, {})
        
        total = read_file(paths.get('total', ''))
        gpu = read_file(paths.get('gpu', ''))
        cpu = read_file(paths.get('cpu', ''))
        
        return PowerMeasurement(
            timestamp=time.time(),
            total_power_mw=total or 0.0,
            gpu_power_mw=gpu,
            cpu_power_mw=cpu,
        )


class GPUMonitor:
    """
    Monitor GPU memory and utilization
    
    Supports NVIDIA GPUs via nvidia-smi or pynvml
    """
    
    def __init__(self):
        self.nvml_available = self._init_nvml()
    
    def _init_nvml(self) -> bool:
        """Try to initialize NVML"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.device_count = pynvml.nvmlDeviceGetCount()
            return True
        except:
            return False
    
    def get_memory(self, device_id: int = 0) -> Dict:
        """Get GPU memory usage"""
        if not self.nvml_available:
            return {'used_mb': 0, 'total_mb': 0, 'free_mb': 0}
        
        try:
            handle = self.nvml.nvmlDeviceGetHandleByIndex(device_id)
            info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                'used_mb': info.used / 1024 / 1024,
                'total_mb': info.total / 1024 / 1024,
                'free_mb': info.free / 1024 / 1024,
            }
        except:
            return {'used_mb': 0, 'total_mb': 0, 'free_mb': 0}
    
    def get_utilization(self, device_id: int = 0) -> Dict:
        """Get GPU utilization"""
        if not self.nvml_available:
            return {'gpu_util': 0, 'memory_util': 0}
        
        try:
            handle = self.nvml.nvmlDeviceGetHandleByIndex(device_id)
            util = self.nvml.nvmlDeviceGetUtilizationRates(handle)
            
            return {
                'gpu_util': util.gpu,
                'memory_util': util.memory,
            }
        except:
            return {'gpu_util': 0, 'memory_util': 0}


class PerformanceProfiler:
    """
    Main performance profiler for Privacy-TMO
    
    Profiles:
    1. Local inference latency
    2. Cloud inference latency
    3. Sensitivity classification latency
    4. Query decomposition latency
    5. Response aggregation latency
    6. LoRA adapter switching
    7. Memory usage
    8. Power consumption (Jetson)
    """
    
    def __init__(self, enable_power: bool = True):
        self.measurements: Dict[str, List[LatencyMeasurement]] = {}
        self.memory_snapshots: List[MemorySnapshot] = []
        self.power_measurements: List[PowerMeasurement] = []
        
        self.gpu_monitor = GPUMonitor()
        self.power_monitor = JetsonPowerMonitor() if enable_power else None
        
        self._profiling = False
    
    @contextmanager
    def measure(self, component: str, metadata: Dict = None):
        """
        Context manager for measuring latency
        
        Usage:
            with profiler.measure("local_inference"):
                result = model.generate(prompt)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            
            measurement = LatencyMeasurement(
                component=component,
                duration_ms=duration_ms,
                timestamp=time.time(),
                metadata=metadata or {},
            )
            
            if component not in self.measurements:
                self.measurements[component] = []
            self.measurements[component].append(measurement)
    
    def snapshot_memory(self):
        """Take memory snapshot"""
        gpu_mem = self.gpu_monitor.get_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(),
            ram_used_mb=psutil.virtual_memory().used / 1024 / 1024,
            ram_total_mb=psutil.virtual_memory().total / 1024 / 1024,
            gpu_used_mb=gpu_mem['used_mb'],
            gpu_total_mb=gpu_mem['total_mb'],
        )
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def snapshot_power(self):
        """Take power snapshot (Jetson only)"""
        if self.power_monitor:
            measurement = self.power_monitor.read_power()
            self.power_measurements.append(measurement)
            return measurement
        return None
    
    def start_continuous_monitoring(self, interval_ms: int = 100):
        """Start background monitoring thread"""
        import threading
        
        self._profiling = True
        
        def monitor_loop():
            while self._profiling:
                self.snapshot_memory()
                self.snapshot_power()
                time.sleep(interval_ms / 1000)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_continuous_monitoring(self):
        """Stop background monitoring"""
        self._profiling = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)
    
    def profile_inference_pipeline(
        self,
        query: str,
        sensitivity_fn: Callable,
        decompose_fn: Callable,
        local_inference_fn: Optional[Callable] = None,
        cloud_inference_fn: Optional[Callable] = None,
        aggregate_fn: Callable = None,
    ) -> Dict:
        """
        Profile complete inference pipeline
        
        Args:
            query: Input query
            sensitivity_fn: Function to classify sensitivity
            decompose_fn: Function to decompose query
            local_inference_fn: Local LLM inference function
            cloud_inference_fn: Cloud LLM inference function
            aggregate_fn: Response aggregation function
        
        Returns:
            Dict with timing breakdown
        """
        results = {
            'query': query,
            'timings': {},
        }
        
        # Take initial memory snapshot
        self.snapshot_memory()
        
        # 1. Sensitivity classification
        with self.measure("sensitivity_classification"):
            sensitivity = sensitivity_fn(query)
        results['timings']['sensitivity_ms'] = self.measurements["sensitivity_classification"][-1].duration_ms
        
        # 2. Query decomposition
        with self.measure("query_decomposition"):
            decomposed = decompose_fn(query)
        results['timings']['decomposition_ms'] = self.measurements["query_decomposition"][-1].duration_ms
        
        # 3. Local inference (if needed)
        local_response = None
        if local_inference_fn and hasattr(decomposed, 'local_query') and decomposed.local_query:
            with self.measure("local_inference"):
                local_response, _ = local_inference_fn(decomposed.local_query)
            results['timings']['local_inference_ms'] = self.measurements["local_inference"][-1].duration_ms
        
        # 4. Cloud inference (if allowed)
        cloud_response = None
        if cloud_inference_fn and hasattr(decomposed, 'cloud_query') and decomposed.cloud_query:
            with self.measure("cloud_inference"):
                cloud_response, _ = cloud_inference_fn(decomposed.cloud_query)
            results['timings']['cloud_inference_ms'] = self.measurements["cloud_inference"][-1].duration_ms
        
        # 5. Response aggregation
        if aggregate_fn and (local_response or cloud_response):
            with self.measure("aggregation"):
                final = aggregate_fn(decomposed, local_response, cloud_response)
            results['timings']['aggregation_ms'] = self.measurements["aggregation"][-1].duration_ms
        
        # Take final memory snapshot
        final_mem = self.snapshot_memory()
        results['memory'] = {
            'ram_used_mb': final_mem.ram_used_mb,
            'gpu_used_mb': final_mem.gpu_used_mb,
        }
        
        # Calculate total
        results['timings']['total_ms'] = sum(results['timings'].values())
        
        return results
    
    def profile_lora_switching(
        self,
        adapter_names: List[str],
        load_adapter_fn: Callable,
        num_iterations: int = 10,
    ) -> Dict:
        """
        Profile LoRA adapter switching overhead
        
        Args:
            adapter_names: List of adapter names to switch between
            load_adapter_fn: Function to load adapter
            num_iterations: Number of switch iterations
        """
        results = {
            'adapters': adapter_names,
            'iterations': num_iterations,
            'switch_times': [],
        }
        
        for _ in range(num_iterations):
            for adapter in adapter_names:
                with self.measure(f"lora_switch_{adapter}"):
                    load_adapter_fn(adapter)
                
                results['switch_times'].append(
                    self.measurements[f"lora_switch_{adapter}"][-1].duration_ms
                )
        
        results['mean_switch_ms'] = sum(results['switch_times']) / len(results['switch_times'])
        results['max_switch_ms'] = max(results['switch_times'])
        
        return results
    
    def get_summary(self) -> Dict:
        """Get profiling summary"""
        summary = {
            'components': {},
            'memory': {},
            'power': {},
        }
        
        # Component latencies
        for component, measurements in self.measurements.items():
            durations = [m.duration_ms for m in measurements]
            summary['components'][component] = {
                'mean_ms': sum(durations) / len(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'count': len(durations),
            }
        
        # Memory
        if self.memory_snapshots:
            ram_used = [s.ram_used_mb for s in self.memory_snapshots]
            summary['memory'] = {
                'ram_mean_mb': sum(ram_used) / len(ram_used),
                'ram_max_mb': max(ram_used),
            }
            
            gpu_used = [s.gpu_used_mb for s in self.memory_snapshots if s.gpu_used_mb]
            if gpu_used:
                summary['memory']['gpu_mean_mb'] = sum(gpu_used) / len(gpu_used)
                summary['memory']['gpu_max_mb'] = max(gpu_used)
        
        # Power
        if self.power_measurements:
            powers = [p.total_power_mw for p in self.power_measurements]
            summary['power'] = {
                'mean_mw': sum(powers) / len(powers),
                'max_mw': max(powers),
            }
        
        return summary
    
    def generate_report(self) -> str:
        """Generate profiling report"""
        summary = self.get_summary()
        
        lines = ["=" * 60]
        lines.append("PERFORMANCE PROFILING REPORT")
        lines.append("=" * 60)
        
        # Latency breakdown
        lines.append("\n📊 LATENCY BREAKDOWN")
        lines.append("-" * 60)
        lines.append(f"{'Component':<30} {'Mean':>10} {'Min':>10} {'Max':>10}")
        lines.append("-" * 60)
        
        for component, stats in summary['components'].items():
            lines.append(
                f"{component:<30} {stats['mean_ms']:>10.2f} "
                f"{stats['min_ms']:>10.2f} {stats['max_ms']:>10.2f}"
            )
        
        # Memory
        if summary['memory']:
            lines.append("\n💾 MEMORY USAGE")
            lines.append("-" * 60)
            lines.append(f"RAM Mean: {summary['memory'].get('ram_mean_mb', 0):.1f} MB")
            lines.append(f"RAM Max: {summary['memory'].get('ram_max_mb', 0):.1f} MB")
            if 'gpu_mean_mb' in summary['memory']:
                lines.append(f"GPU Mean: {summary['memory']['gpu_mean_mb']:.1f} MB")
                lines.append(f"GPU Max: {summary['memory']['gpu_max_mb']:.1f} MB")
        
        # Power
        if summary['power']:
            lines.append("\n⚡ POWER CONSUMPTION")
            lines.append("-" * 60)
            lines.append(f"Mean Power: {summary['power']['mean_mw']:.1f} mW")
            lines.append(f"Max Power: {summary['power']['max_mw']:.1f} mW")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
    
    def save_results(self, filepath: str):
        """Save profiling results to JSON"""
        data = {
            'summary': self.get_summary(),
            'measurements': {
                k: [{'duration_ms': m.duration_ms, 'timestamp': m.timestamp}
                    for m in v]
                for k, v in self.measurements.items()
            },
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Profiling results saved to: {filepath}")


if __name__ == "__main__":
    print("🧪 Testing Performance Profiler\n")
    
    profiler = PerformanceProfiler(enable_power=False)
    
    # Simulate profiling
    for i in range(5):
        # Simulate sensitivity classification
        with profiler.measure("sensitivity_classification"):
            time.sleep(0.01)  # 10ms
        
        # Simulate local inference
        with profiler.measure("local_inference"):
            time.sleep(0.1)  # 100ms
        
        # Simulate cloud inference
        with profiler.measure("cloud_inference"):
            time.sleep(0.05)  # 50ms
        
        profiler.snapshot_memory()
    
    # Generate report
    print(profiler.generate_report())
    
    # Save results
    profiler.save_results("./profile_results/test_run.json")
    
    print("\n✅ Performance Profiler ready!")
