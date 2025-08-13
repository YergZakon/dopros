"""
GPU/CPU Manager with comprehensive device handling and error recovery
"""

import logging
import platform
import subprocess
import time
import threading
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Union, Tuple
import psutil

try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class GPUManager:
    """Singleton GPU/CPU manager with comprehensive device handling"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        
        # Device state
        self.current_device = None
        self.available_devices = []
        self.device_capabilities = {}
        self.fallback_to_cpu = False
        self.cuda_compatible = False
        self.mps_available = False
        self.directml_available = False
        
        # Memory tracking
        self.memory_history = []
        self.memory_limit = None
        
        # Error tracking
        self.error_counts = {}
        self.last_errors = {}
        
        # Initialize NVML if available
        self._init_nvml()
        
        # Detect available devices
        self._detect_devices()
        
        self.logger.info("GPUManager initialized")
    
    @classmethod
    def get_instance(cls) -> 'GPUManager':
        """Get singleton instance"""
        return cls()
    
    def _init_nvml(self):
        """Initialize NVIDIA ML if available"""
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
                self.logger.info("NVIDIA ML initialized successfully")
            except Exception as e:
                self.nvml_initialized = False
                self.logger.warning(f"Failed to initialize NVIDIA ML: {e}")
        else:
            self.nvml_initialized = False
            self.logger.info("NVIDIA ML not available")
    
    def _detect_devices(self):
        """Detect all available compute devices"""
        self.available_devices = ['cpu']
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, using CPU only")
            self.current_device = torch.device('cpu') if TORCH_AVAILABLE else 'cpu'
            return
        
        # Check CUDA availability
        if torch.cuda.is_available():
            try:
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                
                self.logger.info(f"CUDA available: version {cuda_version}, {device_count} devices")
                
                for i in range(device_count):
                    device_name = f"cuda:{i}"
                    self.available_devices.append(device_name)
                    
                    # Get device capabilities
                    props = torch.cuda.get_device_properties(i)
                    self.device_capabilities[device_name] = {
                        'name': props.name,
                        'compute_capability': f"{props.major}.{props.minor}",
                        'total_memory': props.total_memory,
                        'multiprocessor_count': props.multi_processor_count
                    }
                
                self.cuda_compatible = self._check_cuda_compatibility()
                
            except Exception as e:
                self.logger.error(f"Error detecting CUDA devices: {e}")
                self.cuda_compatible = False
        
        # Check MPS (Metal Performance Shaders) on macOS
        if platform.system() == 'Darwin' and hasattr(torch.backends, 'mps'):
            try:
                if torch.backends.mps.is_available():
                    self.available_devices.append('mps')
                    self.mps_available = True
                    self.logger.info("MPS (Metal) device available")
            except Exception as e:
                self.logger.warning(f"MPS detection failed: {e}")
        
        # Check DirectML on Windows
        if platform.system() == 'Windows':
            try:
                # DirectML detection logic would go here
                # For now, we'll just log that it's being checked
                self.logger.info("Checking for DirectML support...")
                # self.directml_available = self._check_directml()
            except Exception as e:
                self.logger.warning(f"DirectML detection failed: {e}")
        
        self.logger.info(f"Available devices: {self.available_devices}")
    
    def _check_cuda_compatibility(self) -> bool:
        """Check CUDA and PyTorch compatibility"""
        if not torch.cuda.is_available():
            return False
        
        try:
            # Check if we can create a tensor on GPU
            test_tensor = torch.tensor([1.0, 2.0], device='cuda:0')
            result = test_tensor + 1
            del test_tensor, result
            torch.cuda.empty_cache()
            
            self.logger.info("CUDA compatibility test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"CUDA compatibility test failed: {e}")
            return False
    
    def setup_device(self, force_cpu: bool = False, preferred_gpu: int = 0) -> Union[torch.device, str]:
        """
        Setup compute device with fallback strategies
        
        Args:
            force_cpu: Force CPU usage
            preferred_gpu: Preferred GPU index
            
        Returns:
            Selected device
        """
        if force_cpu or not TORCH_AVAILABLE:
            self.current_device = torch.device('cpu') if TORCH_AVAILABLE else 'cpu'
            self.logger.info("Using CPU (forced or PyTorch unavailable)")
            return self.current_device
        
        # Try to use preferred GPU first
        if self.cuda_compatible and f"cuda:{preferred_gpu}" in self.available_devices:
            try:
                device = torch.device(f"cuda:{preferred_gpu}")
                
                # Test device with simple operation
                test_tensor = torch.tensor([1.0], device=device)
                _ = test_tensor * 2
                del test_tensor
                torch.cuda.empty_cache()
                
                self.current_device = device
                self.logger.info(f"Using GPU: cuda:{preferred_gpu}")
                return self.current_device
                
            except Exception as e:
                self.logger.warning(f"Failed to use cuda:{preferred_gpu}: {e}")
                self._record_error(f"cuda:{preferred_gpu}", e)
        
        # Try MPS on macOS
        if self.mps_available:
            try:
                device = torch.device('mps')
                test_tensor = torch.tensor([1.0], device=device)
                _ = test_tensor * 2
                del test_tensor
                
                self.current_device = device
                self.logger.info("Using MPS device")
                return self.current_device
                
            except Exception as e:
                self.logger.warning(f"Failed to use MPS: {e}")
                self._record_error('mps', e)
        
        # Fallback to any available CUDA device
        for device_name in self.available_devices:
            if device_name.startswith('cuda:'):
                try:
                    device = torch.device(device_name)
                    test_tensor = torch.tensor([1.0], device=device)
                    _ = test_tensor * 2
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    self.current_device = device
                    self.logger.info(f"Using GPU fallback: {device_name}")
                    return self.current_device
                    
                except Exception as e:
                    self.logger.warning(f"Failed to use {device_name}: {e}")
                    self._record_error(device_name, e)
        
        # Final fallback to CPU
        self.current_device = torch.device('cpu')
        self.fallback_to_cpu = True
        self.logger.warning("All GPU devices failed, falling back to CPU")
        return self.current_device
    
    def test_cuda_nms(self) -> bool:
        """
        Test torchvision NMS operation for YOLO compatibility
        
        Returns:
            True if NMS works correctly
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self.logger.info("CUDA not available for NMS test")
            return False
        
        try:
            device = torch.device('cuda:0') if 'cuda:0' in self.available_devices else self.current_device
            
            # Create test data for NMS
            boxes = torch.tensor([
                [0, 0, 10, 10],
                [5, 5, 15, 15],
                [20, 20, 30, 30]
            ], dtype=torch.float32, device=device)
            
            scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device=device)
            
            # Test torchvision NMS
            keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
            
            # Verify result
            if len(keep) > 0 and keep.device.type == device.type:
                self.logger.info("CUDA NMS test passed")
                return True
            else:
                self.logger.warning("CUDA NMS test failed - unexpected result")
                return False
                
        except Exception as e:
            self.logger.error(f"CUDA NMS test failed: {e}")
            self._record_error('nms_test', e)
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            'current_device': str(self.current_device) if self.current_device else None,
            'available_devices': self.available_devices,
            'device_capabilities': self.device_capabilities,
            'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'cuda_compatible': self.cuda_compatible,
            'mps_available': self.mps_available,
            'directml_available': self.directml_available,
            'pytorch_version': torch.__version__ if TORCH_AVAILABLE else None,
            'cuda_version': torch.version.cuda if TORCH_AVAILABLE and torch.cuda.is_available() else None,
            'fallback_to_cpu': self.fallback_to_cpu,
            'error_counts': dict(self.error_counts),
            'platform': platform.system(),
            'python_version': platform.python_version()
        }
        
        # Add NVIDIA GPU info if available
        if self.nvml_initialized and torch.cuda.is_available():
            try:
                gpu_info = []
                device_count = nvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode()
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    
                    gpu_info.append({
                        'index': i,
                        'name': name,
                        'memory_total': memory_info.total,
                        'memory_free': memory_info.free,
                        'memory_used': memory_info.used,
                        'temperature': temperature
                    })
                
                info['nvidia_gpus'] = gpu_info
                
            except Exception as e:
                self.logger.warning(f"Failed to get NVIDIA GPU info: {e}")
        
        return info
    
    def get_memory_info(self, device: Optional[str] = None) -> Dict[str, Any]:
        """Get memory usage information"""
        if device is None:
            device = str(self.current_device) if self.current_device else 'cpu'
        
        memory_info = {
            'device': device,
            'timestamp': time.time()
        }
        
        if device.startswith('cuda') and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device_idx = int(device.split(':')[1]) if ':' in device else 0
                
                # PyTorch memory info
                memory_info.update({
                    'allocated': torch.cuda.memory_allocated(device_idx),
                    'reserved': torch.cuda.memory_reserved(device_idx),
                    'max_allocated': torch.cuda.max_memory_allocated(device_idx),
                    'max_reserved': torch.cuda.max_memory_reserved(device_idx)
                })
                
                # NVML memory info
                if self.nvml_initialized:
                    handle = nvml.nvmlDeviceGetHandleByIndex(device_idx)
                    nvml_memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_info.update({
                        'total_memory': nvml_memory.total,
                        'free_memory': nvml_memory.free,
                        'used_memory': nvml_memory.used,
                        'utilization': (nvml_memory.used / nvml_memory.total) * 100
                    })
                    
            except Exception as e:
                self.logger.warning(f"Failed to get CUDA memory info: {e}")
        
        # System memory info
        memory_info['system_memory'] = {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'used': psutil.virtual_memory().used,
            'percent': psutil.virtual_memory().percent
        }
        
        return memory_info
    
    def clear_cache(self, device: Optional[str] = None):
        """Clear GPU memory cache"""
        if device is None:
            device = str(self.current_device) if self.current_device else 'cpu'
        
        if device.startswith('cuda') and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                if ':' in device:
                    device_idx = int(device.split(':')[1])
                    with torch.cuda.device(device_idx):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                else:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                self.logger.info(f"Cleared cache for {device}")
                
            except Exception as e:
                self.logger.warning(f"Failed to clear cache for {device}: {e}")
        
        elif device == 'mps' and TORCH_AVAILABLE and self.mps_available:
            try:
                # MPS doesn't have explicit cache clearing
                # but we can try to free up some memory
                import gc
                gc.collect()
                self.logger.info("Cleared MPS memory")
                
            except Exception as e:
                self.logger.warning(f"Failed to clear MPS memory: {e}")
    
    def benchmark_device(self, device: Optional[str] = None, iterations: int = 100) -> Dict[str, float]:
        """Benchmark device performance"""
        if device is None:
            device = str(self.current_device) if self.current_device else 'cpu'
        
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        try:
            torch_device = torch.device(device)
            
            # Benchmark parameters
            matrix_size = 1024
            
            # Create test tensors
            a = torch.randn(matrix_size, matrix_size, device=torch_device)
            b = torch.randn(matrix_size, matrix_size, device=torch_device)
            
            # Warmup
            for _ in range(10):
                _ = torch.matmul(a, b)
            
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            
            # Benchmark matrix multiplication
            start_time = time.time()
            for _ in range(iterations):
                result = torch.matmul(a, b)
            
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / iterations
            operations_per_second = iterations / total_time
            
            # Cleanup
            del a, b, result
            self.clear_cache(device)
            
            benchmark_results = {
                'device': device,
                'matrix_size': matrix_size,
                'iterations': iterations,
                'total_time': total_time,
                'average_time': avg_time,
                'operations_per_second': operations_per_second,
                'gflops': (2 * matrix_size ** 3 * iterations) / (total_time * 1e9)
            }
            
            self.logger.info(f"Benchmark completed for {device}: {operations_per_second:.2f} ops/sec")
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Benchmark failed for {device}: {e}")
            return {'error': str(e)}
    
    def auto_select_device(self) -> Union[torch.device, str]:
        """Automatically select the best available device"""
        if not TORCH_AVAILABLE:
            return 'cpu'
        
        # Run benchmarks on available devices
        device_scores = {}
        
        for device_name in self.available_devices:
            if device_name == 'cpu':
                continue
                
            try:
                benchmark = self.benchmark_device(device_name, iterations=10)
                if 'error' not in benchmark:
                    # Score based on operations per second and memory available
                    score = benchmark.get('operations_per_second', 0)
                    
                    # Add memory bonus
                    memory_info = self.get_memory_info(device_name)
                    if 'free_memory' in memory_info:
                        memory_bonus = memory_info['free_memory'] / 1e9  # GB
                        score += memory_bonus * 10
                    
                    device_scores[device_name] = score
                    
            except Exception as e:
                self.logger.warning(f"Auto-select benchmark failed for {device_name}: {e}")
        
        if device_scores:
            best_device = max(device_scores, key=device_scores.get)
            self.current_device = torch.device(best_device)
            self.logger.info(f"Auto-selected device: {best_device} (score: {device_scores[best_device]:.2f})")
            return self.current_device
        else:
            # Fallback to CPU
            self.current_device = torch.device('cpu')
            self.logger.info("Auto-selected device: CPU (no GPU available)")
            return self.current_device
    
    @contextmanager
    def device_context(self, device: str):
        """Context manager for temporary device switching"""
        if not TORCH_AVAILABLE:
            yield 'cpu'
            return
        
        original_device = self.current_device
        try:
            temp_device = torch.device(device)
            self.current_device = temp_device
            self.logger.debug(f"Switched to temporary device: {device}")
            yield temp_device
        finally:
            self.current_device = original_device
            self.logger.debug(f"Restored original device: {original_device}")
    
    @contextmanager
    def memory_limit(self, limit_mb: int):
        """Context manager for GPU memory limitation"""
        if not TORCH_AVAILABLE or not str(self.current_device).startswith('cuda'):
            yield
            return
        
        original_limit = self.memory_limit
        try:
            self.memory_limit = limit_mb * 1024 * 1024  # Convert to bytes
            
            # Set PyTorch memory fraction if possible
            device_idx = int(str(self.current_device).split(':')[1]) if ':' in str(self.current_device) else 0
            torch.cuda.set_per_process_memory_fraction(
                limit_mb / (torch.cuda.get_device_properties(device_idx).total_memory / 1024 / 1024),
                device_idx
            )
            
            self.logger.info(f"Set memory limit: {limit_mb} MB")
            yield
            
        finally:
            self.memory_limit = original_limit
            if original_limit is None:
                torch.cuda.set_per_process_memory_fraction(1.0, device_idx)
            self.logger.debug("Restored original memory limit")
    
    def _record_error(self, device: str, error: Exception):
        """Record device-specific errors for tracking"""
        error_key = f"{device}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = {
            'timestamp': time.time(),
            'message': str(error),
            'count': self.error_counts[error_key]
        }
        
        # Auto-fallback logic for specific errors
        if isinstance(error, RuntimeError):
            error_msg = str(error).lower()
            
            # CUDA out of memory
            if 'out of memory' in error_msg:
                self.logger.warning(f"CUDA OOM detected for {device}, clearing cache")
                self.clear_cache(device)
                
            # torchvision::nms error
            elif 'torchvision::nms' in error_msg or 'nms' in error_msg:
                self.logger.warning(f"NMS error detected for {device}, may need CPU fallback")
                
            # CUDNN errors
            elif 'cudnn' in error_msg:
                self.logger.warning(f"CUDNN error detected for {device}")
    
    def handle_yolo_errors(self, func, *args, **kwargs):
        """
        Wrapper for YOLO operations with automatic error handling
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises exception after fallback attempts
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                
                if 'torchvision::nms' in error_msg:
                    self.logger.warning(f"YOLO NMS error (attempt {attempt + 1}): {e}")
                    
                    if attempt < max_retries - 1:
                        # Try clearing cache and retrying
                        self.clear_cache()
                        continue
                    else:
                        # Final attempt: force CPU
                        self.logger.error("Forcing CPU fallback for YOLO NMS")
                        original_device = self.current_device
                        self.current_device = torch.device('cpu')
                        try:
                            result = func(*args, **kwargs)
                            self.logger.warning("YOLO operation succeeded on CPU fallback")
                            return result
                        finally:
                            self.current_device = original_device
                
                elif 'out of memory' in error_msg:
                    self.logger.warning(f"CUDA OOM in YOLO (attempt {attempt + 1}): {e}")
                    self.clear_cache()
                    
                    if attempt < max_retries - 1:
                        continue
                    else:
                        # Force CPU fallback
                        original_device = self.current_device
                        self.current_device = torch.device('cpu')
                        try:
                            result = func(*args, **kwargs)
                            self.logger.warning("YOLO operation succeeded on CPU after OOM")
                            return result
                        finally:
                            self.current_device = original_device
                
                else:
                    # Other RuntimeError, re-raise immediately
                    self._record_error(str(self.current_device), e)
                    raise
                    
            except Exception as e:
                # Non-RuntimeError exceptions, re-raise immediately
                self._record_error(str(self.current_device), e)
                raise
        
        # If we get here, all retries failed
        raise RuntimeError(f"YOLO operation failed after {max_retries} attempts")


# Convenience function to get singleton instance
def get_gpu_manager() -> GPUManager:
    """Get GPU manager singleton instance"""
    return GPUManager.get_instance()