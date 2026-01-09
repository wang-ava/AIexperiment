"""
GPU 支持工具模块
自动检测 GPU 并选择使用 NumPy 或 CuPy
如果检测到 CUDA 和 CuPy，则使用 GPU 加速
否则回退到 CPU 上的 NumPy
"""
import os

# 尝试导入 CuPy
try:
    import cupy as cp
    import cupy.cuda.runtime as cuda_runtime
    
    # 检查是否有可用的 GPU
    try:
        device_count = cuda_runtime.getDeviceCount()
        if device_count > 0:
            # 设置默认设备
            cp.cuda.Device(0).use()
            USE_GPU = True
            print(f"✓ 检测到 {device_count} 个 GPU，使用 CuPy 进行 GPU 加速")
            print(f"  当前 GPU: {cp.cuda.Device().id}")
            print(f"  GPU 名称: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}")
        else:
            USE_GPU = False
            print("⚠ 未检测到可用的 GPU，使用 CPU (NumPy)")
    except Exception as e:
        USE_GPU = False
        print(f"⚠ GPU 检测失败: {e}，使用 CPU (NumPy)")
    
except ImportError:
    USE_GPU = False
    print("⚠ CuPy 未安装，使用 CPU (NumPy)")
    print("  提示: 要启用 GPU 加速，请安装 CuPy: pip install cupy-cuda11x (根据你的 CUDA 版本)")

# 导入 NumPy 作为备用
import numpy as np

# 根据 GPU 可用性选择数组库
if USE_GPU:
    xp = cp  # 使用 CuPy
    ARRAY_MODULE = 'cupy'
else:
    xp = np  # 使用 NumPy
    ARRAY_MODULE = 'numpy'


def get_array_module():
    """
    获取当前使用的数组模块（NumPy 或 CuPy）
    
    返回:
        xp: numpy 或 cupy 模块
    """
    return xp


def to_cpu(array):
    """
    将数组从 GPU 传输到 CPU（如果是 CuPy 数组）
    如果是 NumPy 数组，直接返回
    
    参数:
        array: numpy 或 cupy 数组
    
    返回:
        numpy 数组
    """
    if USE_GPU and hasattr(array, 'get'):  # CuPy 数组有 get() 方法
        return array.get()
    return array


def to_gpu(array):
    """
    将数组从 CPU 传输到 GPU（如果使用 GPU）
    如果使用 CPU，直接返回
    
    参数:
        array: numpy 数组
    
    返回:
        numpy 或 cupy 数组
    """
    if USE_GPU:
        return cp.asarray(array)
    return array


def asnumpy(array):
    """
    将数组转换为 NumPy 数组（兼容 CuPy 的 asnumpy）
    
    参数:
        array: numpy 或 cupy 数组
    
    返回:
        numpy 数组
    """
    if USE_GPU and hasattr(array, 'get'):
        return array.get()
    return np.asarray(array)


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
        'array_module': ARRAY_MODULE
    }
    
    if USE_GPU:
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            info['gpu_name'] = props['name'].decode('utf-8')
            info['gpu_memory'] = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024**3)  # GB
            info['device_count'] = cp.cuda.runtime.getDeviceCount()
        except:
            pass
    
    return info


def clear_gpu_memory():
    """
    清理GPU内存（如果使用GPU）
    建议在训练循环中的关键位置调用，特别是在处理大batch之后
    """
    if USE_GPU:
        try:
            import gc
            # 清理Python的垃圾收集器
            gc.collect()
            # 清理CuPy的内存池
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            pinned_mempool.free_all_blocks()
        except Exception as e:
            # 如果清理失败，不影响程序运行
            pass


def get_gpu_memory_usage():
    """
    获取当前GPU内存使用情况（MB）
    
    返回:
        dict: {'used': 已使用内存(MB), 'total': 总内存(MB), 'free': 空闲内存(MB)}
    """
    if not USE_GPU:
        return None
    
    try:
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
        free_bytes = total_bytes - used_bytes
        
        return {
            'used': used_bytes / (1024**2),  # MB
            'total': total_bytes / (1024**2),  # MB
            'free': free_bytes / (1024**2)  # MB
        }
    except:
        return None


# 导出常用的数组操作，使其可以直接使用
# 这样代码中可以使用 xp.array(), xp.zeros() 等，自动适配 NumPy 或 CuPy
__all__ = [
    'xp', 'USE_GPU', 'ARRAY_MODULE',
    'get_array_module', 'to_cpu', 'to_gpu', 'asnumpy',
    'is_gpu_available', 'get_device_info', 'clear_gpu_memory', 'get_gpu_memory_usage'
]

