import torch;
import psutil;
import GPUtil;

class GPUMonitor:
    """
    Monitor GPU usage and performance metrics.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU Available: {self.gpu_name}")
            print(f"Number of GPUs: {self.gpu_count}")
        else:
            print("No GPU available, using CPU")
    
    def get_gpu_memory_info(self):
        """Get current GPU memory usage."""
        if not self.gpu_available:
            return None
        
        memory_info = {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
            'free': (torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_allocated()) / 1024**2,      # MB
            'total': torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        }
        return memory_info
    
    def get_gpu_utilization(self):
        """Get GPU utilization using GPUtil."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'gpu_util': gpu.load * 100,  # Percentage
                    'memory_util': gpu.memoryUtil * 100,  # Percentage
                    'temperature': gpu.temperature,  # Celsius
                    'power_draw': getattr(gpu, 'powerDraw', None),  # Watts
                    'power_limit': getattr(gpu, 'powerLimit', None)  # Watts
                }
        except:
            pass
        return None
    
    def get_system_info(self):
        """Get system resource usage."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / 1024**3
        }
    
    def log_to_tensorboard(self, writer, step):
        """Log GPU metrics to TensorBoard."""
        if self.gpu_available:
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                writer.add_scalar('GPU/memory_allocated_mb', memory_info['allocated'], step)
                writer.add_scalar('GPU/memory_reserved_mb', memory_info['reserved'], step)
                writer.add_scalar('GPU/memory_free_mb', memory_info['free'], step)
                writer.add_scalar('GPU/memory_total_mb', memory_info['total'], step)
            
            gpu_util = self.get_gpu_utilization()
            if gpu_util:
                writer.add_scalar('GPU/utilization_percent', gpu_util['gpu_util'], step)
                writer.add_scalar('GPU/memory_utilization_percent', gpu_util['memory_util'], step)
                writer.add_scalar('GPU/temperature_celsius', gpu_util['temperature'], step)
                if gpu_util['power_draw']:
                    writer.add_scalar('GPU/power_draw_watts', gpu_util['power_draw'], step)
        
        # Log system metrics
        sys_info = self.get_system_info()
        writer.add_scalar('System/cpu_percent', sys_info['cpu_percent'], step)
        writer.add_scalar('System/memory_percent', sys_info['memory_percent'], step)
    
    def print_current_status(self):
        """Print current GPU and system status."""
        print("\n=== GPU & System Status ===")
        
        if self.gpu_available:
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                print(f"GPU Memory: {memory_info['allocated']:.1f}/{memory_info['total']:.1f} MB "
                      f"({memory_info['allocated']/memory_info['total']*100:.1f}%)")
            
            gpu_util = self.get_gpu_utilization()
            if gpu_util:
                print(f"GPU Utilization: {gpu_util['gpu_util']:.1f}%")
                print(f"GPU Temperature: {gpu_util['temperature']}°C")
        
        sys_info = self.get_system_info()
        print(f"CPU Usage: {sys_info['cpu_percent']:.1f}%")
        print(f"System Memory: {sys_info['memory_percent']:.1f}%")
        print("=" * 30)