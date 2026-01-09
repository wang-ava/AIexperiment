#!/bin/bash

# Fashion-MNIST项目上传到GitHub脚本
# 使用前请先配置Git用户信息和GitHub仓库URL

echo "=========================================="
echo "Fashion-MNIST 项目上传到GitHub"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "run.py" ]; then
    echo "错误: 请在github/目录下运行此脚本"
    exit 1
fi

# 检查git用户配置
if [ -z "$(git config user.name)" ] || [ -z "$(git config user.email)" ]; then
    echo "⚠️  Git用户信息未配置"
    echo ""
    read -p "请输入你的Git用户名: " git_name
    read -p "请输入你的Git邮箱: " git_email
    git config user.name "$git_name"
    git config user.email "$git_email"
    echo "✓ Git用户信息已配置"
    echo ""
fi

# 检查是否已有提交
if [ -z "$(git log --oneline 2>/dev/null)" ]; then
    echo "正在创建初始提交..."
    git commit -m "Initial commit: Fashion-MNIST neural network implementations with NumPy

- Implemented 7 neural network models: MLP, CNN, LeNet-5, ResNet, Wide ResNet, DenseNet-BC, Capsule Network
- Pure NumPy implementation without deep learning frameworks
- Interactive training script with menu system
- Automatic report generation for each model
- GPU support via CuPy (optional)
- Comprehensive documentation"
    echo "✓ 初始提交已创建"
    echo ""
fi

# 检查远程仓库
if [ -z "$(git remote -v)" ]; then
    echo "⚠️  未配置远程仓库"
    echo ""
    echo "请先在GitHub上创建新仓库，然后："
    echo "1. 复制仓库URL（例如: https://github.com/username/repo.git）"
    echo ""
    read -p "请输入GitHub仓库URL: " repo_url
    
    if [ -z "$repo_url" ]; then
        echo "错误: 未提供仓库URL"
        exit 1
    fi
    
    git remote add origin "$repo_url"
    echo "✓ 远程仓库已添加"
    echo ""
fi

# 重命名分支为main（如果当前是master）
current_branch=$(git branch --show-current)
if [ "$current_branch" = "master" ]; then
    git branch -M main
    echo "✓ 分支已重命名为main"
    echo ""
fi

# 推送到GitHub
echo "正在推送到GitHub..."
echo ""
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 代码已成功上传到GitHub！"
    echo "=========================================="
    echo ""
    echo "访问你的GitHub仓库查看代码："
    git remote get-url origin
else
    echo ""
    echo "=========================================="
    echo "✗ 推送失败"
    echo "=========================================="
    echo ""
    echo "可能的原因："
    echo "1. 需要配置SSH密钥或使用Personal Access Token"
    echo "2. 仓库URL不正确"
    echo "3. 网络连接问题"
    echo ""
    echo "请查看 UPLOAD_TO_GITHUB.md 获取详细说明"
fi

