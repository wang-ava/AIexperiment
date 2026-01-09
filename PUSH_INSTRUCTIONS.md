# 推送代码到GitHub的说明

代码已经准备好，但需要配置认证才能推送到GitHub。有两种方式：

## 方式1: 配置SSH密钥（推荐）

### 步骤1: 检查是否已有SSH密钥

```bash
ls -la ~/.ssh
```

如果看到 `id_rsa` 和 `id_rsa.pub` 文件，说明已有密钥。

### 步骤2: 如果没有SSH密钥，生成新的

```bash
ssh-keygen -t ed25519 -C "vitality@mail.ustc.edu.cn"
```

按回车使用默认路径，可以设置密码或直接回车。

### 步骤3: 复制公钥

```bash
cat ~/.ssh/id_ed25519.pub
# 或者如果是rsa密钥
cat ~/.ssh/id_rsa.pub
```

### 步骤4: 添加到GitHub

1. 登录GitHub
2. 点击右上角头像 → Settings
3. 左侧菜单选择 "SSH and GPG keys"
4. 点击 "New SSH key"
5. Title填写：`autodl-server`（或任意名称）
6. Key粘贴刚才复制的公钥内容
7. 点击 "Add SSH key"

### 步骤5: 测试SSH连接

```bash
ssh -T git@github.com
```

如果看到 "Hi wang-ava! You've successfully authenticated..." 说明配置成功。

### 步骤6: 切换回SSH URL并推送

```bash
cd /root/autodl-tmp/aiexperiment/github
git remote set-url origin git@github.com:wang-ava/AIexperiment.git
git push -u origin main
```

---

## 方式2: 使用Personal Access Token（HTTPS）

### 步骤1: 创建Personal Access Token

1. 登录GitHub
2. 点击右上角头像 → Settings
3. 左侧菜单最下方选择 "Developer settings"
4. 选择 "Personal access tokens" → "Tokens (classic)"
5. 点击 "Generate new token" → "Generate new token (classic)"
6. Note填写：`AIexperiment upload`
7. 选择过期时间（建议选择较长时间）
8. 勾选权限：至少需要 `repo` 权限
9. 点击 "Generate token"
10. **重要**：复制生成的token（只显示一次）

### 步骤2: 使用Token推送

```bash
cd /root/autodl-tmp/aiexperiment/github
git remote set-url origin https://github.com/wang-ava/AIexperiment.git
git push -u origin main
```

当提示输入用户名时，输入：`wang-ava`
当提示输入密码时，**粘贴刚才复制的token**（不是GitHub密码）

---

## 当前状态

✅ Git仓库已初始化
✅ 所有文件已添加
✅ 初始提交已创建
✅ 远程仓库已配置：`git@github.com:wang-ava/AIexperiment.git`
✅ 分支已重命名为 `main`

**只需要配置认证后执行：**
```bash
cd /root/autodl-tmp/aiexperiment/github
git push -u origin main
```

---

## 如果遇到问题

### 问题1: Permission denied (publickey)
- 解决：按照方式1配置SSH密钥

### 问题2: 用户名密码认证失败
- 解决：使用Personal Access Token而不是密码（方式2）

### 问题3: 仓库不存在
- 解决：先在GitHub上创建仓库 `AIexperiment`

### 问题4: 需要强制推送
如果远程仓库已有内容，需要先拉取：
```bash
git pull origin main --allow-unrelated-histories
# 解决可能的冲突后
git push -u origin main
```

