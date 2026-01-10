"""
数据增强工具模块
提供用于Fashion-MNIST数据集的数据增强功能
"""
import torch
import torch.nn.functional as F
import numpy as np
import random


class DataAugmentation:
    """数据增强类，支持多种增强策略"""
    
    def __init__(self, use_augmentation=True):
        """
        初始化数据增强
        
        参数:
            use_augmentation: 是否启用数据增强
        """
        self.use_augmentation = use_augmentation
    
    def random_rotation(self, images, max_angle=15):
        """
        随机旋转图像
        
        参数:
            images: 输入图像张量 (batch, channels, height, width)
            max_angle: 最大旋转角度（度）
        
        返回:
            旋转后的图像
        """
        if not self.use_augmentation:
            return images
        
        batch_size = images.shape[0]
        device = images.device
        
        # 生成随机角度
        angles = torch.FloatTensor(batch_size).uniform_(-max_angle, max_angle).to(device)
        
        # 对每张图像应用旋转
        rotated = []
        for i in range(batch_size):
            angle_rad = angles[i] * 3.14159265359 / 180.0  # pi
            # 创建旋转矩阵
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            
            # 使用仿射变换进行旋转
            theta = torch.stack([
                torch.stack([cos_a, -sin_a, torch.tensor(0.0, device=device)]),
                torch.stack([sin_a, cos_a, torch.tensor(0.0, device=device)])
            ]).unsqueeze(0)
            
            grid = F.affine_grid(theta, images[i:i+1].shape, align_corners=False)
            rotated_img = F.grid_sample(images[i:i+1], grid, align_corners=False)
            rotated.append(rotated_img)
        
        return torch.cat(rotated, dim=0)
    
    def random_translation(self, images, max_shift=2):
        """
        随机平移图像
        
        参数:
            images: 输入图像张量 (batch, channels, height, width)
            max_shift: 最大平移像素数
        
        返回:
            平移后的图像
        """
        if not self.use_augmentation:
            return images
        
        batch_size = images.shape[0]
        device = images.device
        
        # 生成随机平移量
        tx = torch.FloatTensor(batch_size).uniform_(-max_shift, max_shift).to(device) / 28.0
        ty = torch.FloatTensor(batch_size).uniform_(-max_shift, max_shift).to(device) / 28.0
        
        # 对每张图像应用平移
        translated = []
        for i in range(batch_size):
            theta = torch.tensor([[1, 0, tx[i].item()],
                                 [0, 1, ty[i].item()]], dtype=torch.float32).unsqueeze(0).to(device)
            
            grid = F.affine_grid(theta, images[i:i+1].shape, align_corners=False)
            translated_img = F.grid_sample(images[i:i+1], grid, align_corners=False, mode='bilinear', padding_mode='zeros')
            translated.append(translated_img)
        
        return torch.cat(translated, dim=0)
    
    def random_crop(self, images, padding=2):
        """
        随机裁剪图像（先padding再crop）
        
        参数:
            images: 输入图像张量 (batch, channels, height, width)
            padding: padding大小
        
        返回:
            裁剪后的图像
        """
        if not self.use_augmentation:
            return images
        
        # 先进行padding
        padded = F.pad(images, (padding, padding, padding, padding), mode='reflect')
        
        # 随机裁剪
        batch_size = images.shape[0]
        h, w = images.shape[2], images.shape[3]
        
        crops = []
        for i in range(batch_size):
            top = random.randint(0, 2 * padding)
            left = random.randint(0, 2 * padding)
            cropped = padded[i:i+1, :, top:top+h, left:left+w]
            crops.append(cropped)
        
        return torch.cat(crops, dim=0)
    
    def random_flip_horizontal(self, images, p=0.5):
        """
        随机水平翻转
        
        参数:
            images: 输入图像张量 (batch, channels, height, width)
            p: 翻转概率
        
        返回:
            翻转后的图像
        """
        if not self.use_augmentation or random.random() > p:
            return images
        
        return torch.flip(images, dims=[3])
    
    def random_erase(self, images, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        """
        Random Erasing数据增强
        
        参数:
            images: 输入图像张量 (batch, channels, height, width)
            p: 应用概率
            scale: 擦除区域面积比例范围
            ratio: 擦除区域宽高比范围
            value: 填充值
        
        返回:
            擦除后的图像
        """
        if not self.use_augmentation or random.random() > p:
            return images
        
        batch_size, c, h, w = images.shape
        erased_images = images.clone()
        
        for i in range(batch_size):
            if random.random() > p:
                continue
            
            # 计算擦除区域
            area = h * w
            target_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])
            
            erasing_h = int(round(np.sqrt(target_area * aspect_ratio)))
            erasing_w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if erasing_h < h and erasing_w < w:
                top = random.randint(0, h - erasing_h)
                left = random.randint(0, w - erasing_w)
                erased_images[i, :, top:top+erasing_h, left:left+erasing_w] = value
        
        return erased_images
    
    def augment(self, images, use_rotation=True, use_translation=True, use_crop=True, 
                use_flip=True, use_erase=True):
        """
        应用所有数据增强
        
        参数:
            images: 输入图像张量 (batch, channels, height, width)
            use_rotation: 是否使用旋转
            use_translation: 是否使用平移
            use_crop: 是否使用裁剪
            use_flip: 是否使用翻转
            use_erase: 是否使用随机擦除
        
        返回:
            增强后的图像
        """
        if not self.use_augmentation:
            return images
        
        augmented = images
        
        if use_crop:
            augmented = self.random_crop(augmented, padding=2)
        
        if use_rotation:
            augmented = self.random_rotation(augmented, max_angle=10)
        
        if use_translation:
            augmented = self.random_translation(augmented, max_shift=2)
        
        if use_flip:
            augmented = self.random_flip_horizontal(augmented, p=0.5)
        
        if use_erase:
            augmented = self.random_erase(augmented, p=0.25, scale=(0.02, 0.15), value=0)
        
        # 确保值在[0, 1]范围内
        augmented = torch.clamp(augmented, 0.0, 1.0)
        
        return augmented


def label_smoothing_loss(pred, target, num_classes=10, smoothing=0.1):
    """
    Label Smoothing损失函数
    
    参数:
        pred: 预测logits (batch_size, num_classes)
        target: 真实标签 (batch_size,)
        num_classes: 类别数
        smoothing: 平滑系数
    
    返回:
        平滑后的交叉熵损失
    """
    log_probs = F.log_softmax(pred, dim=1)
    
    # 创建平滑后的标签
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - smoothing)
    
    return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

