#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: otto
"""

# ============================================================
# COMPREHENSIVE ML/DL HARDWARE BENCHMARK SCRIPT
# ============================================================

import torch
import platform
import psutil
import subprocess
import time
import numpy as np
from datetime import datetime
import sys
import os

# Try to import optional libraries
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Note: Install GPUtil for more GPU details: pip install gputil")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from torch.utils.benchmark import Timer
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


class HardwareBenchmark:
    """Comprehensive hardware benchmark for ML/DL workloads."""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def print_header(self, title):
        """Print formatted section header."""
        print("\n" + "="*70)
        print(f"{title:^70}")
        print("="*70)
    
    def print_subheader(self, title):
        """Print formatted subsection header."""
        print(f"\n{title}")
        print("-" * 70)
    
    # ============================================================
    # SYSTEM INFORMATION
    # ============================================================
    
    def get_system_info(self):
        """Get basic system information."""
        self.print_header("SYSTEM INFORMATION")
        
        info = {
            'OS': platform.system(),
            'OS Version': platform.version(),
            'Architecture': platform.machine(),
            'Processor': platform.processor(),
            'Python Version': sys.version.split()[0],
            'Hostname': platform.node()
        }
        
        for key, value in info.items():
            print(f"{key:<20}: {value}")
        
        self.results['system'] = info
        return info
    
    # ============================================================
    # CPU INFORMATION
    # ============================================================
    
    def get_cpu_info(self):
        """Get detailed CPU information."""
        self.print_header("CPU INFORMATION")
        
        cpu_info = {
            'Physical Cores': psutil.cpu_count(logical=False),
            'Total Cores': psutil.cpu_count(logical=True),
            'Max Frequency': f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() else "N/A",
            'Current Frequency': f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "N/A",
            'CPU Usage': f"{psutil.cpu_percent(interval=1):.1f}%"
        }
        
        for key, value in cpu_info.items():
            print(f"{key:<20}: {value}")
        
        # Per-core usage
        print(f"\nPer-Core Usage:")
        for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
            print(f"  Core {i:<2}: {percentage:>5.1f}%")
        
        self.results['cpu'] = cpu_info
        return cpu_info
    
    # ============================================================
    # MEMORY INFORMATION
    # ============================================================
    
    def get_memory_info(self):
        """Get memory information."""
        self.print_header("MEMORY INFORMATION")
        
        svmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            'Total RAM': self.bytes_to_gb(svmem.total),
            'Available RAM': self.bytes_to_gb(svmem.available),
            'Used RAM': self.bytes_to_gb(svmem.used),
            'RAM Usage': f"{svmem.percent:.1f}%",
            'Total Swap': self.bytes_to_gb(swap.total),
            'Used Swap': self.bytes_to_gb(swap.used),
            'Swap Usage': f"{swap.percent:.1f}%"
        }
        
        for key, value in memory_info.items():
            print(f"{key:<20}: {value}")
        
        self.results['memory'] = memory_info
        return memory_info
    
    # ============================================================
    # GPU INFORMATION
    # ============================================================
    
    def get_gpu_info(self):
        """Get GPU information."""
        self.print_header("GPU INFORMATION")
        
        gpu_info = {
            'CUDA Available': torch.cuda.is_available(),
            'CUDA Version': torch.version.cuda if torch.cuda.is_available() else "N/A",
            'cuDNN Version': torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A",
            'Number of GPUs': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        print(f"CUDA Available       : {gpu_info['CUDA Available']}")
        print(f"CUDA Version         : {gpu_info['CUDA Version']}")
        print(f"cuDNN Version        : {gpu_info['cuDNN Version']}")
        print(f"Number of GPUs       : {gpu_info['Number of GPUs']}")
        
        if torch.cuda.is_available():
            gpu_devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    'index': i,
                    'name': props.name,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'total_memory': self.bytes_to_gb(props.total_memory),
                    'multiprocessors': props.multi_processor_count
                }
                gpu_devices.append(device_info)
                
                self.print_subheader(f"GPU {i}: {props.name}")
                print(f"  Compute Capability : {device_info['compute_capability']}")
                print(f"  Total Memory       : {device_info['total_memory']}")
                print(f"  Multiprocessors    : {device_info['multiprocessors']}")
                
                # Current memory usage
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                print(f"  Memory Allocated   : {self.bytes_to_gb(allocated)}")
                print(f"  Memory Reserved    : {self.bytes_to_gb(reserved)}")
                
                # Additional details with GPUtil
                if GPUTIL_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if i < len(gpus):
                            gpu = gpus[i]
                            print(f"  GPU Load           : {gpu.load*100:.1f}%")
                            print(f"  GPU Temperature    : {gpu.temperature}°C")
                            print(f"  GPU Memory Used    : {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
                    except:
                        pass
            
            gpu_info['devices'] = gpu_devices
        
        self.results['gpu'] = gpu_info
        return gpu_info
    
    # ============================================================
    # PYTORCH INFORMATION
    # ============================================================
    
    def get_pytorch_info(self):
        """Get PyTorch configuration."""
        self.print_header("PYTORCH INFORMATION")
        
        pytorch_info = {
            'PyTorch Version': torch.__version__,
            'CUDA Available': torch.cuda.is_available(),
            'cuDNN Enabled': torch.backends.cudnn.enabled,
            'cuDNN Version': torch.backends.cudnn.version() if torch.backends.cudnn.enabled else "N/A",
            'Number of Threads': torch.get_num_threads(),
            'MPS Available (macOS)': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        for key, value in pytorch_info.items():
            print(f"{key:<25}: {value}")
        
        # Build configuration
        print(f"\nBuild Configuration:")
        print(f"  Built with CUDA      : {torch.version.cuda is not None}")
        if hasattr(torch, '__config__'):
            print(f"  Debug build          : {torch.__config__.show().find('DEBUG') != -1}")
        
        self.results['pytorch'] = pytorch_info
        return pytorch_info
    
    # ============================================================
    # CPU BENCHMARKS
    # ============================================================
    
    def benchmark_cpu(self):
        """Benchmark CPU performance for ML operations."""
        self.print_header("CPU BENCHMARKS")
        
        results = {}
        
        # Matrix multiplication
        self.print_subheader("Matrix Multiplication (CPU)")
        sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
        
        for size in sizes:
            n, m = size
            a = torch.randn(n, m)
            b = torch.randn(m, n)
            
            start = time.time()
            for _ in range(10):
                c = torch.mm(a, b)
            elapsed = (time.time() - start) / 10
            
            gflops = (2 * n * m * n) / (elapsed * 1e9)
            print(f"  {n}x{m} @ {m}x{n}: {elapsed*1000:.2f}ms ({gflops:.2f} GFLOPS)")
            results[f'matmul_{n}x{m}'] = {'time_ms': elapsed*1000, 'gflops': gflops}
        
        # Convolution
        self.print_subheader("2D Convolution (CPU)")
        batch_sizes = [1, 8, 32]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224)
            conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            
            start = time.time()
            for _ in range(10):
                y = conv(x)
            elapsed = (time.time() - start) / 10
            
            print(f"  Batch {batch_size}: {elapsed*1000:.2f}ms")
            results[f'conv2d_batch_{batch_size}'] = {'time_ms': elapsed*1000}
        
        self.results['cpu_benchmark'] = results
        return results
    
    # ============================================================
    # GPU BENCHMARKS
    # ============================================================
    
    def benchmark_gpu(self):
        """Benchmark GPU performance for ML operations."""
        if not torch.cuda.is_available():
            print("\n⚠️  CUDA not available. Skipping GPU benchmarks.")
            return None
        
        self.print_header("GPU BENCHMARKS")
        
        device = torch.device('cuda:0')
        results = {}
        
        # Matrix multiplication
        self.print_subheader("Matrix Multiplication (GPU)")
        sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000), (4000, 4000)]
        
        for size in sizes:
            n, m = size
            a = torch.randn(n, m, device=device)
            b = torch.randn(m, n, device=device)
            
            # Warmup
            for _ in range(5):
                c = torch.mm(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(20):
                c = torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / 20
            
            gflops = (2 * n * m * n) / (elapsed * 1e9)
            print(f"  {n}x{m} @ {m}x{n}: {elapsed*1000:.2f}ms ({gflops:.2f} GFLOPS)")
            results[f'matmul_{n}x{m}'] = {'time_ms': elapsed*1000, 'gflops': gflops}
        
        # Convolution
        self.print_subheader("2D Convolution (GPU)")
        batch_sizes = [1, 8, 32, 64]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1).to(device)
            
            # Warmup
            for _ in range(5):
                y = conv(x)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(20):
                y = conv(x)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / 20
            
            print(f"  Batch {batch_size}: {elapsed*1000:.2f}ms")
            results[f'conv2d_batch_{batch_size}'] = {'time_ms': elapsed*1000}
        
        # Mixed precision (FP16)
        self.print_subheader("Mixed Precision (FP16) Performance")
        
        n = 2000
        a_fp32 = torch.randn(n, n, device=device)
        b_fp32 = torch.randn(n, n, device=device)
        a_fp16 = a_fp32.half()
        b_fp16 = b_fp32.half()
        
        # FP32
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            c = torch.mm(a_fp32, b_fp32)
        torch.cuda.synchronize()
        fp32_time = (time.time() - start) / 20
        
        # FP16
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            c = torch.mm(a_fp16, b_fp16)
        torch.cuda.synchronize()
        fp16_time = (time.time() - start) / 20
        
        speedup = fp32_time / fp16_time
        print(f"  FP32: {fp32_time*1000:.2f}ms")
        print(f"  FP16: {fp16_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results['mixed_precision'] = {
            'fp32_ms': fp32_time*1000,
            'fp16_ms': fp16_time*1000,
            'speedup': speedup
        }
        
        # Memory bandwidth test
        self.print_subheader("GPU Memory Bandwidth")
        
        size = 100_000_000  # 100M elements
        data = torch.randn(size, device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            result = data * 2.0
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10
        
        bytes_transferred = size * 4 * 2  # 4 bytes per float, read and write
        bandwidth_gbps = (bytes_transferred / elapsed) / 1e9
        
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        results['memory_bandwidth_gbps'] = bandwidth_gbps
        
        self.results['gpu_benchmark'] = results
        return results
    
    # ============================================================
    # ML-SPECIFIC BENCHMARKS
    # ============================================================
    
    def benchmark_ml_operations(self):
        """Benchmark common ML operations."""
        self.print_header("ML OPERATIONS BENCHMARK")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = {}
        
        # ResNet-like block
        self.print_subheader(f"ResNet Block Forward Pass ({device})")
        
        batch_sizes = [1, 8, 32]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 64, 56, 56, device=device)
            
            block = torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64)
            ).to(device)
            
            # Warmup
            if torch.cuda.is_available():
                for _ in range(5):
                    y = block(x)
                torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            iterations = 50
            for _ in range(iterations):
                y = block(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start) / iterations
            print(f"  Batch {batch_size}: {elapsed*1000:.2f}ms")
            results[f'resnet_block_batch_{batch_size}'] = elapsed*1000
        
        # Transformer attention
        self.print_subheader(f"Transformer Attention ({device})")
        
        seq_lengths = [128, 512, 1024]
        d_model = 512
        num_heads = 8
        
        for seq_len in seq_lengths:
            x = torch.randn(1, seq_len, d_model, device=device)
            
            attention = torch.nn.MultiheadAttention(
                d_model, num_heads, batch_first=True
            ).to(device)
            
            # Warmup
            if torch.cuda.is_available():
                for _ in range(5):
                    y, _ = attention(x, x, x)
                torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            iterations = 20
            for _ in range(iterations):
                y, _ = attention(x, x, x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start) / iterations
            print(f"  Seq Length {seq_len}: {elapsed*1000:.2f}ms")
            results[f'attention_seqlen_{seq_len}'] = elapsed*1000
        
        self.results['ml_operations'] = results
        return results
    
    # ============================================================
    # DATA LOADING BENCHMARK
    # ============================================================
    
    def benchmark_data_loading(self):
        """Benchmark data loading speed."""
        self.print_header("DATA LOADING BENCHMARK")
        
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy dataset
        num_samples = 10000
        data = torch.randn(num_samples, 3, 224, 224)
        labels = torch.randint(0, 1000, (num_samples,))
        dataset = TensorDataset(data, labels)
        
        results = {}
        
        # Test different num_workers
        self.print_subheader("DataLoader with Different Workers")
        
        num_workers_list = [0, 2, 4, 8]
        batch_size = 32
        
        for num_workers in num_workers_list:
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            start = time.time()
            for batch in dataloader:
                pass
            elapsed = time.time() - start
            
            samples_per_sec = num_samples / elapsed
            print(f"  {num_workers} workers: {elapsed:.2f}s ({samples_per_sec:.0f} samples/s)")
            results[f'workers_{num_workers}'] = {
                'time_s': elapsed,
                'samples_per_sec': samples_per_sec
            }
        
        self.results['data_loading'] = results
        return results
    
    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================
    
    @staticmethod
    def bytes_to_gb(bytes_val):
        """Convert bytes to GB."""
        return f"{bytes_val / (1024**3):.2f} GB"
    
    def generate_summary(self):
        """Generate summary of all benchmarks."""
        self.print_header("BENCHMARK SUMMARY")
        
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\nTotal Benchmark Duration: {duration:.2f} seconds")
        
        # Hardware Summary
        print(f"\n{'Hardware Configuration':-^70}")
        print(f"CPU Cores: {self.results['cpu']['Total Cores']}")
        print(f"RAM: {self.results['memory']['Total RAM']}")
        
        if self.results['gpu']['CUDA Available']:
            print(f"GPU: {self.results['gpu']['devices'][0]['name']}")
            print(f"GPU Memory: {self.results['gpu']['devices'][0]['total_memory']}")
        else:
            print(f"GPU: None (CPU only)")
        
        # Performance Summary
        if 'gpu_benchmark' in self.results and self.results['gpu_benchmark']:
            print(f"\n{'GPU Performance Highlights':-^70}")
            gpu_bench = self.results['gpu_benchmark']
            
            if 'matmul_2000x2000' in gpu_bench:
                matmul_perf = gpu_bench['matmul_2000x2000']['gflops']
                print(f"Matrix Multiplication (2000x2000): {matmul_perf:.2f} GFLOPS")
            
            if 'memory_bandwidth_gbps' in gpu_bench:
                print(f"Memory Bandwidth: {gpu_bench['memory_bandwidth_gbps']:.2f} GB/s")
            
            if 'mixed_precision' in gpu_bench:
                speedup = gpu_bench['mixed_precision']['speedup']
                print(f"FP16 Speedup: {speedup:.2f}x")
        
        # Recommendations
        print(f"\n{'Recommendations':-^70}")
        
        total_cores = self.results['cpu']['Total Cores']
        if total_cores >= 16:
            print("✓ CPU: Excellent for parallel data loading")
        elif total_cores >= 8:
            print("✓ CPU: Good for most ML workloads")
        else:
            print("⚠ CPU: Consider upgrading for faster data preprocessing")
        
        ram_gb = float(self.results['memory']['Total RAM'].split()[0])
        if ram_gb >= 32:
            print("✓ RAM: Excellent for large datasets")
        elif ram_gb >= 16:
            print("✓ RAM: Good for medium-sized datasets")
        else:
            print("⚠ RAM: May limit batch size and dataset caching")
        
        if self.results['gpu']['CUDA Available']:
            gpu_mem_gb = float(self.results['gpu']['devices'][0]['total_memory'].split()[0])
            if gpu_mem_gb >= 16:
                print("✓ GPU Memory: Excellent for large models")
            elif gpu_mem_gb >= 8:
                print("✓ GPU Memory: Good for most models")
            else:
                print("⚠ GPU Memory: May limit model size and batch size")
        else:
            print("⚠ GPU: Consider getting a CUDA-capable GPU for deep learning")
    
    def save_results(self, filename='hardware_benchmark_results.txt'):
        """Save results to file."""
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HARDWARE BENCHMARK RESULTS\n")
            f.write(f"Generated: {self.start_time}\n")
            f.write("="*70 + "\n\n")
            
            for key, value in self.results.items():
                f.write(f"\n{key.upper()}:\n")
                f.write(str(value) + "\n")
        
        print(f"\n✓ Results saved to {filename}")
    
    # ============================================================
    # RUN ALL BENCHMARKS
    # ============================================================
    
    def run_all(self, skip_heavy=False):
        """Run all benchmarks."""
        print("\n" + "="*70)
        print(f"{'HARDWARE BENCHMARK FOR ML/DEEP LEARNING':^70}")
        print(f"{'Started at: ' + self.start_time.strftime('%Y-%m-%d %H:%M:%S'):^70}")
        print("="*70)
        
        # System info
        self.get_system_info()
        self.get_cpu_info()
        self.get_memory_info()
        self.get_gpu_info()
        self.get_pytorch_info()
        
        # Benchmarks
        if not skip_heavy:
            self.benchmark_cpu()
            self.benchmark_gpu()
            self.benchmark_ml_operations()
            self.benchmark_data_loading()
        
        # Summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        return self.results


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n🚀 Starting Hardware Benchmark...")
    print("This may take a few minutes...\n")
    
    # Create benchmark instance
    benchmark = HardwareBenchmark()
    
    # Run all benchmarks
    # Set skip_heavy=True to skip time-consuming benchmarks
    results = benchmark.run_all(skip_heavy=False)
    
    print("\n✅ Benchmark completed!")
    print("\nResults have been saved to 'hardware_benchmark_results.txt'")