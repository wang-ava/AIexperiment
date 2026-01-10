"""
Capsule Network (CapsNet) - PyTorch版本
论文: "Dynamic Routing Between Capsules" (Sara Sabour, Geoffrey Hinton et al., 2017)

关键特性:
- Capsule替代神经元，向量表示更丰富
- Dynamic routing algorithm
- Squash激活函数
- Margin loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gpu_utils import get_device, to_cpu, is_gpu_available, get_gpu_memory_usage, clear_gpu_memory
from utils import load_fashion_mnist, create_mini_batches, get_class_name, generate_training_report, set_random_seed


class PrimaryCapsule(nn.Module):
    """Primary Capsule层"""
    
    def __init__(self, in_channels, capsule_dim=8, num_capsules=32):
        super(PrimaryCapsule, self).__init__()
        self.capsule_dim = capsule_dim
        self.num_capsules = num_capsules
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim, kernel_size=9, stride=2, padding=0)
    
    def squash(self, vectors):
        """Squash激活函数"""
        squared_norm = (vectors ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vectors = vectors / (torch.sqrt(squared_norm + 1e-8))
        return scale * unit_vectors
    
    def forward(self, x):
        # x: (batch, in_channels, height, width)
        # 卷积输出: (batch, num_capsules * capsule_dim, out_h, out_w)
        output = self.conv(x)
        
        batch_size = output.size(0)
        _, _, out_h, out_w = output.shape
        
        # Reshape为capsule格式: (batch, num_capsules, out_h, out_w, capsule_dim)
        output = output.view(batch_size, self.num_capsules, self.capsule_dim, out_h, out_w)
        output = output.permute(0, 1, 3, 4, 2).contiguous()
        output = output.view(batch_size, self.num_capsules * out_h * out_w, self.capsule_dim)
        
        # 应用squash激活
        output = self.squash(output)
        
        return output


class DigitCapsule(nn.Module):
    """Digit Capsule层 - 使用dynamic routing"""
    
    def __init__(self, num_input_capsules, input_capsule_dim, num_output_capsules=10, output_capsule_dim=16, num_routing=3):
        super(DigitCapsule, self).__init__()
        self.num_input_capsules = num_input_capsules
        self.input_capsule_dim = input_capsule_dim
        self.num_output_capsules = num_output_capsules
        self.output_capsule_dim = output_capsule_dim
        self.num_routing = num_routing
        
        # 变换矩阵: (num_input_capsules, input_capsule_dim, num_output_capsules, output_capsule_dim)
        self.W = nn.Parameter(torch.randn(num_input_capsules, input_capsule_dim, num_output_capsules, output_capsule_dim) * 0.01)
    
    def squash(self, vectors):
        """Squash激活函数"""
        squared_norm = (vectors ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vectors = vectors / (torch.sqrt(squared_norm + 1e-8))
        return scale * unit_vectors
    
    def forward(self, input_capsules):
        """
        Dynamic Routing算法
        input_capsules: (batch, num_input_capsules, input_capsule_dim)
        输出: (batch, num_output_capsules, output_capsule_dim)
        """
        batch_size = input_capsules.size(0)
        
        # 计算预测向量 u_hat
        # input_capsules: (batch, num_input, input_dim)
        # W: (num_input, input_dim, num_output, output_dim)
        # u_hat: (batch, num_input, num_output, output_dim)
        # 修正einsum：bid表示(batch, input, input_dim)，idjo表示(input, input_dim, output, output_dim)
        u_hat = torch.einsum('bid,idjo->bijo', input_capsules, self.W)
        
        # Dynamic Routing
        # 初始化routing logits b为0
        b = torch.zeros(batch_size, self.num_input_capsules, self.num_output_capsules, 1, device=input_capsules.device)
        
        for iteration in range(self.num_routing):
            # Softmax计算routing coefficients c
            c = F.softmax(b, dim=2)  # (batch, num_input, num_output, 1)
            
            # 加权求和: s = sum(c * u_hat)
            s = (c * u_hat).sum(dim=1)  # (batch, num_output, output_dim)
            
            # Squash激活
            v = self.squash(s)  # (batch, num_output, output_dim)
            
            # 更新routing logits (除了最后一次迭代)
            if iteration < self.num_routing - 1:
                # b = b + u_hat · v
                v_expanded = v.unsqueeze(1)  # (batch, 1, num_output, output_dim)
                agreement = (u_hat * v_expanded).sum(dim=-1, keepdim=True)  # (batch, num_input, num_output, 1)
                b = b + agreement
        
        return v
    

class CapsuleNetwork(nn.Module):
    """完整的Capsule Network"""
    
    def __init__(self, learning_rate=0.001):
        super(CapsuleNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.device = get_device()
        
        # 根据GPU显存调整模型规模
        if is_gpu_available():
            mem_info = get_gpu_memory_usage()
            if mem_info and mem_info['free'] > 80000:
                conv1_channels = 256
                num_capsules = 32
            elif mem_info and mem_info['free'] > 50000:
                conv1_channels = 256
                num_capsules = 32
            else:
                conv1_channels = 128
                num_capsules = 16
        else:
            conv1_channels = 128
            num_capsules = 16
        
        # 第一层：标准卷积层
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=9, stride=1, padding=0)
        # 移除inplace操作以避免潜在问题
        self.relu = nn.ReLU()
        
        # Primary Capsule层
        self.primary_caps = PrimaryCapsule(in_channels=conv1_channels, capsule_dim=8, num_capsules=num_capsules)
        
        # 计算primary capsule的输出数量
        # 输入28x28, conv1输出20x20, primary_caps输出6x6
        num_primary_capsules = num_capsules * 6 * 6
        
        # Digit Capsule层
        self.digit_caps = DigitCapsule(
            num_input_capsules=num_primary_capsules,
            input_capsule_dim=8,
            num_output_capsules=10,
            output_capsule_dim=16,
            num_routing=3
        )
        
        # 初始化权重
        self._initialize_weights()
        self.to(self.device)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # x: (batch, 1, 28, 28)
        # 第一层卷积 + ReLU
        x = self.relu(self.conv1(x))  # (batch, conv1_channels, 20, 20)
        
        # Primary Capsule层
        primary_output = self.primary_caps(x)  # (batch, num_primary_capsules, 8)
        
        # Digit Capsule层 (dynamic routing)
        digit_output = self.digit_caps(primary_output)  # (batch, 10, 16)
        
        return digit_output
    
    def margin_loss(self, v, y, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """Margin Loss"""
        # v: (batch, 10, 16)
        # y: (batch,)
        
        # 计算capsule长度
        v_length = torch.sqrt((v ** 2).sum(dim=-1))  # (batch, 10)
        
        # One-hot编码
        y_one_hot = torch.zeros(y.size(0), 10, device=v.device)
        y_one_hot.scatter_(1, y.view(-1, 1), 1.0)
        
        # Margin loss
        loss_present = y_one_hot * torch.clamp(m_plus - v_length, min=0.0) ** 2
        loss_absent = lambda_ * (1 - y_one_hot) * torch.clamp(v_length - m_minus, min=0.0) ** 2
        
        loss = (loss_present + loss_absent).sum(dim=1).mean()
        
        return loss, v_length
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
        """训练模型"""
        X_train_img = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
        X_test_img = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
        
        X_train_tensor = torch.FloatTensor(X_train_img).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_img).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # 添加学习率调度器（更激进的衰减）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
        
        history = {
            'train_acc': [],
            'test_acc': [],
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        self.train()
        
        for epoch in range(epochs):
            batches = create_mini_batches(X_train, y_train, batch_size)
            epoch_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in batches:
                X_batch_img = X_batch.reshape(-1, 1, 28, 28).astype(np.float32)
                X_batch_tensor = torch.FloatTensor(X_batch_img).to(self.device)
                y_batch_tensor = torch.LongTensor(y_batch).to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                digit_output = self.forward(X_batch_tensor)
                
                # 计算margin loss
                loss, v_length = self.margin_loss(digit_output, y_batch_tensor)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # 评估
            train_acc = self.evaluate(X_train_tensor[:5000], y_train_tensor[:5000])
            test_acc = self.evaluate(X_test_tensor, y_test_tensor)
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            # 更新学习率调度器
            scheduler.step(test_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - '
                  f'Train Accuracy: {train_acc:.4f}, '
                  f'Test Accuracy: {test_acc:.4f} - '
                  f'LR: {current_lr:.6f}')
            
            if (epoch + 1) % 5 == 0:
                clear_gpu_memory()
        
        return history
    
    def predict(self, X):
        """预测"""
        self.eval()
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                X = X.cpu().numpy()
            if X.ndim == 2:
                X_img = X.reshape(-1, 1, 28, 28).astype(np.float32)
            else:
                X_img = X.astype(np.float32)
            X_tensor = torch.FloatTensor(X_img).to(self.device)
            digit_output = self.forward(X_tensor)
        
            # Capsule长度作为类别概率
            v_length = torch.sqrt((digit_output ** 2).sum(dim=-1))  # (batch, 10)
            _, predictions = torch.max(v_length, 1)
        
            return predictions.cpu().numpy()
    
    def evaluate(self, X, y):
        """评估模型"""
        self.eval()
        with torch.no_grad():
            digit_output = self.forward(X)
            v_length = torch.sqrt((digit_output ** 2).sum(dim=-1))  # (batch, 10)
            _, predictions = torch.max(v_length, 1)
            accuracy = (predictions == y).float().mean().item()
        return accuracy


def main():
    """主函数：训练Capsule Network模型"""
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("Capsule Network (CapsNet) - Fashion-MNIST分类 (PyTorch版本)")
    print("=" * 70)
    
    set_random_seed(42)
    
    print("\n加载Fashion-MNIST数据集...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    print("\n创建Capsule Network模型...")
    print("网络特点:")
    print("  - Capsule: 用向量替代标量神经元")
    print("  - Dynamic Routing: 迭代路由算法")
    print("  - Squash激活函数")
    print("  - Margin Loss")
    model = CapsuleNetwork(learning_rate=0.0005)
    
    if is_gpu_available():
        mem_info = get_gpu_memory_usage()
        if mem_info:
            print(f"\nGPU内存: {mem_info['used']:.1f}MB / {mem_info['total']:.1f}MB (可用: {mem_info['free']:.1f}MB)")
    
    print("\n开始训练...")
    
    if is_gpu_available():
        mem_info = get_gpu_memory_usage()
        if mem_info and mem_info['free'] > 80000:
            batch_size = 128
        elif mem_info and mem_info['free'] > 50000:
            batch_size = 64
        elif mem_info and mem_info['free'] > 20000:
            batch_size = 32
        else:
            batch_size = 16
    else:
        batch_size = 8
    
    print(f"使用batch_size={batch_size}进行训练")
    # 增加训练轮数以达到95%准确率
    history = model.train_model(
        X_train, y_train,
        X_test, y_test,
        epochs=80,
        batch_size=batch_size
    )
    
    print("\n训练完成！正在生成报告...")
    X_train_img = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
    X_test_img = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
    X_train_tensor = torch.FloatTensor(X_train_img).to(model.device)
    y_train_tensor = torch.LongTensor(y_train).to(model.device)
    X_test_tensor = torch.FloatTensor(X_test_img).to(model.device)
    y_test_tensor = torch.LongTensor(y_test).to(model.device)
    
    train_acc = model.evaluate(X_train_tensor[:5000], y_train_tensor[:5000])
    test_acc = model.evaluate(X_test_tensor, y_test_tensor)
    
    training_time = time.time() - start_time
    
    generate_training_report(
        model_name="Capsule Network",
        history=history,
        train_acc=train_acc,
        test_acc=test_acc,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model=model,
        layer_info="Conv -> PrimaryCaps(32x8) -> DigitCaps(10x16) with Dynamic Routing",
        learning_rate=model.learning_rate,
        training_time=training_time
    )


if __name__ == '__main__':
    main()
