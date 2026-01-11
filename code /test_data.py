"""
测试数据加载功能
验证Fashion-MNIST数据集是否能正常加载
"""
from utils import load_fashion_mnist, get_class_name
import numpy as np  # 数据加载测试使用 CPU (NumPy)


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("Fashion-MNIST 数据加载测试")
    print("=" * 60)
    
    try:
        # 加载数据
        print("\n正在加载数据集...")
        X_train, y_train, X_test, y_test = load_fashion_mnist()
        
        # 显示数据集信息
        print("\n✓ 数据加载成功！")
        print("\n数据集统计信息:")
        print("-" * 60)
        print(f"训练集图像: {X_train.shape}")
        print(f"训练集标签: {y_train.shape}")
        print(f"测试集图像: {X_test.shape}")
        print(f"测试集标签: {y_test.shape}")
        
        # 数据类型和范围
        print("\n数据类型和范围:")
        print("-" * 60)
        print(f"图像数据类型: {X_train.dtype}")
        print(f"标签数据类型: {y_train.dtype}")
        print(f"图像值范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"标签值范围: [{y_train.min()}, {y_train.max()}]")
        
        # 类别分布
        print("\n训练集类别分布:")
        print("-" * 60)
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"类别 {label} ({get_class_name(label):15s}): {count:5d} 张图像")
        
        # 显示一些样本
        print("\n随机样本展示:")
        print("-" * 60)
        sample_indices = np.random.choice(len(X_train), 10, replace=False)
        for idx in sample_indices:
            label = y_train[idx]
            print(f"样本 #{idx:5d}: 类别 {label} - {get_class_name(label)}")
        
        # 验证数据完整性
        print("\n数据完整性检查:")
        print("-" * 60)
        
        # 检查是否有NaN或Inf
        has_nan_train = np.isnan(X_train).any()
        has_inf_train = np.isinf(X_train).any()
        has_nan_test = np.isnan(X_test).any()
        has_inf_test = np.isinf(X_test).any()
        
        print(f"训练集包含NaN: {'是' if has_nan_train else '否'}")
        print(f"训练集包含Inf: {'是' if has_inf_train else '否'}")
        print(f"测试集包含NaN: {'是' if has_nan_test else '否'}")
        print(f"测试集包含Inf: {'是' if has_inf_test else '否'}")
        
        # 最终结果
        print("\n" + "=" * 60)
        if not (has_nan_train or has_inf_train or has_nan_test or has_inf_test):
            print("✓ 所有测试通过！数据集准备就绪。")
        else:
            print("✗ 数据集存在问题，请检查！")
        print("=" * 60)
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ 错误: 找不到数据文件")
        print(f"  请确保dataset文件夹包含以下文件:")
        print(f"  - train-images-idx3-ubyte.gz")
        print(f"  - train-labels-idx1-ubyte.gz")
        print(f"  - test-images-idx3-ubyte.gz")
        print(f"  - test-labels-idx1-ubyte.gz")
        print(f"\n  错误详情: {e}")
        return False
        
    except Exception as e:
        print(f"\n✗ 发生未知错误: {e}")
        return False


if __name__ == '__main__':
    success = test_data_loading()
    
    if success:
        print("\n提示: 你现在可以运行以下脚本训练模型:")
        print("  python mlp.py      # 多层感知机")
        print("  python cnn.py      # 卷积神经网络")
        print("  python lenet.py    # LeNet-5")
        print("  python resnet.py   # 残差网络")

