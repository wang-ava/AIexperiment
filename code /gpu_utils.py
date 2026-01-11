"""
PyTorch GPU 支持工具模块
自动检测 GPU 并使用 PyTorch 的设备管理
"""
import torch
import torch.backends.cudnn as cudnn


# 检查GPU是否可用
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

if USE_GPU:
    print(f"✓ 检测到 {torch.cuda.device_count()} 个 GPU，使用 PyTorch 进行 GPU 加速")
    print(f"  当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA 版本: {torch.version.cuda}")
    # 设置cudnn基准测试以提高性能（如果输入大小固定）
    cudnn.benchmark = True
else:
    print("⚠ 未检测到可用的 GPU，使用 CPU (PyTorch)")


def get_device():
    """
    获取当前设备
    
    返回:
        torch.device: 当前使用的设备 (cuda 或 cpu)
    """
    return DEVICE


def to_cpu(tensor):
    """
    将张量从 GPU 传输到 CPU
    
    参数:
        tensor: PyTorch 张量
    
    返回:
        numpy 数组 (在CPU上)
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return tensor


def to_gpu(array, device=None):
    """
    将数组或张量传输到指定设备（默认GPU）
    
    参数:
        array: numpy 数组或 PyTorch 张量
        device: 目标设备，默认使用全局DEVICE
    
    返回:
        PyTorch 张量 (在指定设备上)
    """
    if device is None:
        device = DEVICE
    
    if isinstance(array, torch.Tensor):
        return array.to(device)
    elif isinstance(array, (list, tuple)):
        return [to_gpu(x, device) for x in array]
    else:
        # 假设是numpy数组
        import numpy as np
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(device)
        return torch.tensor(array).to(device)


def asnumpy(tensor):
    """
    将张量转换为 NumPy 数组
    
    参数:
        tensor: PyTorch 张量
    
    返回:
        numpy 数组
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    import numpy as np
    return np.asarray(tensor)


def is_gpu_available():
    """
    检查 GPU 是否可用
    
    返回:
        bool: 如果 GPU 可用返回 True，否则返回 False
    """
    return USE_GPU


def get_device_info():
    """
    获取设备信息
    
    返回:
        dict: 包含设备信息的字典
    """
    info = {
        'using_gpu': USE_GPU,
        'device': str(DEVICE),
        'device_count': torch.cuda.device_count() if USE_GPU else 0
    }
    
    if USE_GPU:
        try:
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        except Exception:
            pass
    
    return info


def clear_gpu_memory():
    """
    清理GPU内存（如果使用GPU）
    建议在训练循环中的关键位置调用，特别是在处理大batch之后
    """
    if USE_GPU:
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def get_gpu_memory_usage():
    """
    获取当前GPU内存使用情况（MB）
    
    返回:
        dict: {'used': 已使用内存(MB), 'total': 总内存(MB), 'free': 空闲内存(MB)}
    """
    if not USE_GPU:
        return None
    
    try:
        used_bytes = torch.cuda.memory_allocated(0)
        reserved_bytes = torch.cuda.memory_reserved(0)
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        
        return {
            'used': used_bytes / (1024**2),  # MB
            'reserved': reserved_bytes / (1024**2),  # MB
            'total': total_bytes / (1024**2),  # MB
            'free': (total_bytes - reserved_bytes) / (1024**2)  # MB
        }
    except Exception:
        return None


# 兼容旧代码：提供xp变量（指向torch，但不推荐直接使用）
# 新代码应该直接使用torch
xp = torch
ARRAY_MODULE = 'pytorch'


def get_array_module():
    """
    获取当前使用的数组模块（PyTorch）
    
    返回:
        torch: PyTorch 模块
    """
    return torch


__all__ = [
    'DEVICE', 'USE_GPU', 'ARRAY_MODULE',
    'get_device', 'to_cpu', 'to_gpu', 'asnumpy',
    'is_gpu_available', 'get_device_info', 'clear_gpu_memory', 'get_gpu_memory_usage',
    'get_array_module', 'xp'
]
