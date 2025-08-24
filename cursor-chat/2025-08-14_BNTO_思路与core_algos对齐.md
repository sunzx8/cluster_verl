## 主题: BNTO 思路与 core_algos.py 实现对齐

### 结论要点
- **Dual变量**: `LinearKLController` 管 beta, `EntropyBudgetController` 管 lambda/B；`compute_policy_loss_dual_game` 使用 `beta_coef` 与 `lambda_coef`。
- **加权熵项**: 代码使用 `w = p(1-p)*|A|*|log p - log p0|`，与论文 `p(1-p)*|A|` 有差异（多了 KL 残差因子）。
- **梯度形式**: 实现等价于式(6)：`adv_base - entropy_penalty - kl_penalty` 后走 PPO-clip。
- **预算与更新**: `EntropyBudgetController.update()` 用每 token 预算 `B_t=B/T` 做 λ 更新；`decay_target()` 用奖励方差和上批预算使用率混合衰减；未实现 EMA-平台判定与 K 步 warm-up 度量；`_update_adaptive_b_0()` 未被调用。
- **KL 控制**: `LinearKLController` 做 β ← [β + αβ(D_kl − D)]_{+}，并支持自适应目标 KL。
- **默认参数**: `gamma` 默认 0.8（会削弱负优势），与论文“γ>1 放大负优势”不一致。

### 后续最小改动建议（仅计划，不改码）
- **改1（匹配式(6)的 w_t）**: `verl/trainer/ppo/core_algos.py` 中 `compute_policy_loss_dual_game()` 约 L1423-L1427：将 `w = prob*(1-prob)*kl_residual*abs(advantages)` 改为 `w = prob*(1-prob)*abs(advantages)`；保留 KL 影响于 `kl_penalty`。
- **改2（γ>1）**: 同函数约 L1403，将 `gamma` 默认值从 `0.8` 调整为 `>1`（如 `1.2`），或从配置读取并在配方中设定。
- **改3（启用 B0 自适应）**: `EntropyBudgetController.update()` 约 L300-L318 计算 `wH_per_token` 后，调用 `self._update_adaptive_b_0(wH_per_token)` 以驱动 `_update_adaptive_b_0()`。
- **改4（平台触发的预算衰减）**: 平台检测与 EMA 应集成在 trainer（如 `verl/trainer/ppo/ray_trainer.py`）里：维护 `ar R_EMA` 与 plateau 计数 `m`，平台时调用 `entropy_ctrl.decay_target(current_step, reward_std=None, usage_ratio=...)`，或新增 `decay_by_plateau(m)` API；warm-up K 步在 trainer 侧统计 `ar H_{warm}` 写回 `B0`。

### 风险与假设
- 这些改动不改变现有训练流程接口；λ/β 的实际更新入口在 trainer，需确保每步将新系数写回 `config.policy_loss.dual_game` 或以状态对象注入。
- 若现有下游脚本依赖当前 `w` 的 KL 因子，改1 会改变学习动态，需要在小学习率下验证。
