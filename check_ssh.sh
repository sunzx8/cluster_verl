#!/bin/bash

# GitHub SSH 端口修复脚本
# 用于解决 SSH 端口 22 被阻塞的问题，配置使用端口 443

set -e

echo "=== GitHub SSH 端口修复脚本 ==="
echo

# 检查 .ssh 目录是否存在
SSH_DIR="$HOME/.ssh"
SSH_CONFIG="$SSH_DIR/config"

echo "1. 检查 SSH 目录..."
if [ ! -d "$SSH_DIR" ]; then
    echo "   创建 .ssh 目录: $SSH_DIR"
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"
else
    echo "   .ssh 目录已存在: $SSH_DIR"
fi

echo

# 备份现有配置文件
echo "2. 备份现有 SSH 配置..."
if [ -f "$SSH_CONFIG" ]; then
    BACKUP_FILE="${SSH_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "   备份现有配置到: $BACKUP_FILE"
    cp "$SSH_CONFIG" "$BACKUP_FILE"
else
    echo "   没有现有配置文件需要备份"
fi

echo

# 检查是否已经配置了 GitHub
echo "3. 检查现有 GitHub 配置..."
GITHUB_CONFIG_EXISTS=false
if [ -f "$SSH_CONFIG" ] && grep -q "Host github.com" "$SSH_CONFIG"; then
    echo "   发现现有 GitHub 配置"
    GITHUB_CONFIG_EXISTS=true
else
    echo "   没有发现 GitHub 配置"
fi

echo

# 添加或更新 GitHub 配置
echo "4. 配置 GitHub SSH..."
if [ "$GITHUB_CONFIG_EXISTS" = true ]; then
    echo "   现有配置已存在，是否要覆盖？(y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # 删除现有的 GitHub 配置
        sed -i '/^Host github\.com$/,/^$/d' "$SSH_CONFIG"
        echo "   已删除现有配置"
    else
        echo "   保持现有配置不变"
        exit 0
    fi
fi

# 添加新的 GitHub 配置
echo "   添加新的 GitHub SSH 配置..."
cat >> "$SSH_CONFIG" << 'EOF'

# GitHub SSH 配置 - 使用端口 443
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
    PreferredAuthentications publickey
    IdentitiesOnly yes

EOF

echo "   配置已添加到 $SSH_CONFIG"

# 设置正确的权限
chmod 600 "$SSH_CONFIG"
echo "   已设置配置文件权限为 600"

echo

# 测试连接
echo "5. 测试 GitHub SSH 连接..."
echo "   正在测试连接到 github.com..."

if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "   ✅ SSH 连接测试成功！"
else
    echo "   ⚠️  SSH 连接测试失败，请检查："
    echo "      - 确保你有有效的 SSH 密钥"
    echo "      - 确保 SSH 密钥已添加到 GitHub 账户"
    echo "      - 网络连接正常"
fi

echo

# 显示当前配置
echo "6. 当前 GitHub SSH 配置："
echo "----------------------------------------"
grep -A 6 "Host github.com" "$SSH_CONFIG" || echo "未找到 GitHub 配置"
echo "----------------------------------------"

echo

# 提供使用说明
echo "7. 使用说明："
echo "   现在你可以正常使用 Git 命令："
echo "   git clone git@github.com:username/repository.git"
echo "   git push origin branch-name"
echo
echo "   如果你的仓库远程 URL 使用的是 ssh.github.com，"
echo "   可以运行以下命令将其改回标准格式："
echo "   git remote set-url origin git@github.com:sunzx8/BNTO_verl.git"

echo
echo "=== 脚本执行完成 ==="
