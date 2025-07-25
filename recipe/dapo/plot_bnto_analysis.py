#!/usr/bin/env python3
"""
BNTO 分析实验绘图脚本
用于可视化 BNTO vs DAPO 的对比实验结果
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd

def load_tensorboard_logs(log_dirs: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    从tensorboard日志目录加载数据
    Args:
        log_dirs: {"方法名": "日志路径"} 的字典
    Returns:
        {"方法名": DataFrame} 的字典
    """
    # 这里简化处理，实际需要根据tensorboard存储格式调整
    # 可以使用 tensorboard.backend.event_processing 或 wandb API
    print("注意：需要根据实际日志格式实现加载逻辑")
    return {}

def plot_entropy_budget_curves(data: Dict[str, pd.DataFrame], save_dir: Path):
    """绘制熵预算相关曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('BNTO Entropy Budget Analysis', fontsize=16)
    
    # 1. wH_per_token vs B_token
    ax1 = axes[0, 0]
    for method, df in data.items():
        if 'analysis/wH_per_token_basic' in df.columns:
            ax1.plot(df['step'], df['analysis/wH_per_token_basic'], 
                    label=f'{method} wH/token', linewidth=2)
        if 'analysis/B_token' in df.columns and method != 'DAPO':
            ax1.plot(df['step'], df['analysis/B_token'], 
                    label=f'{method} B_t', linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Per-token Budget')
    ax1.set_title('Entropy Budget Execution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Lambda & Beta curves
    ax2 = axes[0, 1]
    for method, df in data.items():
        if 'analysis/lambda' in df.columns:
            ax2.plot(df['step'], df['analysis/lambda'], 
                    label=f'{method} λ', linewidth=2)
        if 'analysis/beta' in df.columns:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(df['step'], df['analysis/beta'], 
                         label=f'{method} β', linewidth=2, color='red', alpha=0.7)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Lambda Value')
    ax2.set_title('Dual Variables')
    ax2.legend(loc='upper left')
    if 'ax2_twin' in locals():
        ax2_twin.set_ylabel('Beta Value')
        ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Usage Ratio
    ax3 = axes[1, 0]
    for method, df in data.items():
        if 'analysis/usage_ratio_basic' in df.columns:
            ax3.plot(df['step'], df['analysis/usage_ratio_basic'], 
                    label=f'{method} u_basic', linewidth=2)
        if 'analysis/usage_ratio_kl_gated' in df.columns:
            ax3.plot(df['step'], df['analysis/usage_ratio_kl_gated'], 
                    label=f'{method} u_KL', linewidth=2, linestyle=':')
    
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Target (1.0)')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Usage Ratio')
    ax3.set_title('Budget Usage Efficiency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. KL per token
    ax4 = axes[1, 1]
    for method, df in data.items():
        if 'analysis/KL_per_token' in df.columns:
            ax4.plot(df['step'], df['analysis/KL_per_token'], 
                    label=f'{method}', linewidth=2)
    
    # Add target KL line if available
    ax4.axhline(y=0.04, color='red', linestyle='--', alpha=0.5, label='Target KL (0.04)')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('KL Divergence per Token')
    ax4.set_title('KL Constraint Adherence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'bnto_entropy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_kl_gating_effect(data: Dict[str, pd.DataFrame], save_dir: Path):
    """绘制KL gating效果对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('KL Residual Gating Effect', fontsize=14)
    
    # wH comparison: basic vs KL-gated
    ax1 = axes[0]
    for method, df in data.items():
        if 'KL' in method:  # 只显示有KL gating的方法
            if 'analysis/wH_per_token_basic' in df.columns:
                ax1.plot(df['step'], df['analysis/wH_per_token_basic'], 
                        label=f'{method} Basic', linewidth=2)
            if 'analysis/wH_per_token_kl_gated' in df.columns:
                ax1.plot(df['step'], df['analysis/wH_per_token_kl_gated'], 
                        label=f'{method} KL-gated', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('wH per Token')
    ax1.set_title('Basic vs KL-gated wH')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Usage ratio comparison
    ax2 = axes[1]
    for method, df in data.items():
        if 'analysis/usage_ratio_basic' in df.columns:
            ax2.plot(df['step'], df['analysis/usage_ratio_basic'], 
                    label=f'{method} Basic', linewidth=2)
        if 'analysis/usage_ratio_kl_gated' in df.columns and 'KL' in method:
            ax2.plot(df['step'], df['analysis/usage_ratio_kl_gated'], 
                    label=f'{method} KL-gated', linewidth=2, linestyle='--')
    
    ax2.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Usage Ratio')
    ax2.set_title('Usage Efficiency: Basic vs KL-gated')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'kl_gating_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_reward_vs_budget(data: Dict[str, pd.DataFrame], save_dir: Path):
    """绘制奖励标准差与预算的关系"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for method, df in data.items():
        if 'analysis/reward_std' in df.columns and 'analysis/B_token' in df.columns:
            # 归一化reward_std到[0,1]
            reward_std_norm = df['analysis/reward_std'] / df['analysis/reward_std'].max()
            ax.plot(df['step'], reward_std_norm, label=f'{method} σ_R (norm)', linewidth=2)
            ax.plot(df['step'], df['analysis/B_token'], label=f'{method} B_t', 
                   linewidth=2, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Reward Std vs Budget Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'reward_budget_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='BNTO Analysis Plotting')
    parser.add_argument('--log_dirs', type=str, required=True,
                       help='JSON file containing {"method": "log_dir"} mapping')
    parser.add_argument('--save_dir', type=str, default='./plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Load log directories
    with open(args.log_dirs, 'r') as f:
        log_dirs = json.load(f)
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading tensorboard data...")
    data = load_tensorboard_logs(log_dirs)
    
    if not data:
        print("警告：未加载到数据，请检查load_tensorboard_logs函数实现")
        return
    
    # Generate plots
    print("Generating entropy budget plots...")
    plot_entropy_budget_curves(data, save_dir)
    
    print("Generating KL gating effect plots...")
    plot_kl_gating_effect(data, save_dir)
    
    print("Generating reward-budget correlation plots...")
    plot_reward_vs_budget(data, save_dir)
    
    print(f"All plots saved to {save_dir}")

if __name__ == "__main__":
    main() 