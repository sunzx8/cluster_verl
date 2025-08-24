## 主题: 在 `verl/trainer/ppo/ray_trainer.py` 的 `fit` 中上传四个指标（熵、协方差、W*H、reward）的最小改动方案

### 1) 任务范围
- 仅改动 `verl/trainer/ppo/ray_trainer.py` 的 `fit`，复用已存在张量：`old_log_probs`、`advantages`、`response_mask`、`token_level_rewards`。
- 仅新增一次性指标计算与 `metrics.update(...)`，不改训练逻辑、不引入新依赖。

### 2) 指标链路定义（与 `recipe/BNTO/entropy_ray_trainer.py` 保持口径）
- 记 `logp = old_log_probs`，`prob = exp(logp)`，`A = advantages`，`M = response_mask.bool()`。
- **熵 H**: 使用与 BNTO 方案一致的近似（选中 token 的自信息）`H = -(prob * logp)`；聚合：`H_mean = mean(H[M])`。
- **协方差 Cov(A, logp)**: 定义 `cov = (A - mean(A[M])) * (logp - mean(logp[M]))`；聚合：`cov_mean = mean(cov[M])`。
- **W*H**: `w = prob * (1 - prob) * abs(A)`（不做 KL 门控以保持通用性）；`wH = w * H`；聚合：
  - `wH_sum = sum(wH[M])`
  - `wH_per_token = wH_sum / max(sum(M), 1)`
- **reward**: `R_seq = sum(token_level_rewards * response_mask, dim=-1)`；聚合：`reward_sum_mean = mean(R_seq)`。

### 3) 精确插入点
- 文件：`verl/trainer/ppo/ray_trainer.py`
- 位置：第一处 `fit` 中，优势计算完成之后、更新 critic 之前。
  - 参考当前文件片段：
    - 结束于：`batch = compute_advantage(...);`（大约 L1307）
    - 插入点：紧随其后，加入若干 `torch` 张量运算与 `metrics.update({...})`。

### 4) 指标命名与日志键
- `analysis/entropy_per_token_mean`: `H_mean`
- `analysis/covariance_mean`: `cov_mean`
- `analysis/wH_sum`: `wH_sum`
- `analysis/wH_per_token`: `wH_per_token`
- `analysis/reward_sum_mean`: `reward_sum_mean`

### 5) 伪代码（不做代码编辑，仅供实施时粘贴参考）
```python
resp_mask = batch.batch["response_mask"]
logp = batch.batch["old_log_probs"]
adv = batch.batch["advantages"]
rewards = batch.batch["token_level_rewards"]

valid = resp_mask.bool()
prob = torch.exp(logp)
H = -(prob * logp)

A_mean = adv[valid].mean()
logp_mean = logp[valid].mean()
cov_mean = ((adv - A_mean) * (logp - logp_mean))[valid].mean()

w = prob * (1 - prob) * torch.abs(adv)
wH = w * H
wH_sum = (wH * resp_mask).sum()
token_cnt = resp_mask.sum().clamp_min(1)
wH_per_token = wH_sum / token_cnt

reward_sum_mean = (rewards * resp_mask).sum(dim=-1).mean()

metrics.update({
  "analysis/entropy_per_token_mean": (H * resp_mask).sum() / token_cnt,
  "analysis/covariance_mean": cov_mean,
  "analysis/wH_sum": wH_sum,
  "analysis/wH_per_token": wH_per_token,
  "analysis/reward_sum_mean": reward_sum_mean,
})
```

### 6) 影响评估
- 仅新增 O(N_tokens) 级别张量运算，计算成本可忽略；不改变任何训练分支。
- 依赖的张量均已在 `fit` 里构造，无需新增前置步骤。

### 7) 实施与回滚
- 一次性小改动；若需回滚，删除插入块即可。
