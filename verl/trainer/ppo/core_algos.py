"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig

POLICY_LOSS_REGISTRY = {}
import logging
logger = logging.getLogger(__name__)      # 放在所有代码之前
logger.setLevel(logging.INFO)

def register_policy_loss(name):
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func):
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


ADV_ESTIMATOR_REGISTRY = {}


def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


# ----------------- Dual-Game: Linear KL Controller -----------------
# Implements β ← [β + αβ (D_kl − D)]_{+}


class LinearKLController:
    """Linear (additive) KL controller for Dual-Game RL.

    Args:
        init_beta (float): Initial β coefficient.
        target_kl (float): Desired KL divergence D.
        beta_lr   (float): Learning rate αβ for the dual update.
        adaptive_enabled (bool): Whether to enable adaptive target KL (default: True)
        adaptive_window (int): Window size for adaptive target KL calculation (default: 10)
    """

    def __init__(self, init_beta: float, target_kl: float, beta_lr: float, 
                 adaptive_enabled: bool = True, adaptive_window: int = 10):
        self.value = init_beta
        self.target = target_kl
        self.initial_target = target_kl  # Store initial target for reference
        self.beta_lr = beta_lr
        
        # Adaptive target KL parameters
        self.adaptive_enabled = adaptive_enabled
        self.adaptive_window = adaptive_window
        self.kl_history = []  # Store KL values for adaptive calculation
        self.step_count = 0
        
        logger.info(f"Using linear KL controller: value={self.value}, target={self.target}, beta_lr={self.beta_lr}")
        if self.adaptive_enabled:
            logger.info(f"Adaptive target KL enabled with window size: {self.adaptive_window}")

    def update(self, current_kl: float, n_steps: int = 1):
        """Additive dual update with adaptive target KL.

        β ← [ β + αβ (D_kl − D) ]_{+}
        """
        logger.info(f"LinearKLController: self.value: {self.value}")
        logger.info(f"LinearKLController: self.target: {self.target}")
        logger.info(f"LinearKLController: current_kl: {current_kl}")
        
        # Update adaptive target KL based on history
        if self.adaptive_enabled:
            self._update_adaptive_target_kl(current_kl)
        
        delta = self.beta_lr * (current_kl - self.target)
        logger.info(f"LinearKLController: delta: {delta}")
        self.value = max(1e-6, self.value + delta)  # Add lower bound protection
        logger.info(f"LinearKLController: self.value: {self.value}")
        
        # Increment step counter
        self.step_count += 1

    def _update_adaptive_target_kl(self, current_kl: float):
        """Update adaptive target KL based on historical KL values.
        
        Args:
            current_kl (float): Current KL divergence value
        """
        # Add current KL to history
        self.kl_history.append(current_kl)
        
        # Keep only the last adaptive_window values
        if len(self.kl_history) > self.adaptive_window:
            self.kl_history = self.kl_history[-self.adaptive_window:]
        
        # Update target KL based on average of last adaptive_window steps
        if len(self.kl_history) >= self.adaptive_window:
            avg_kl = sum(self.kl_history) / len(self.kl_history)
            # Set target KL to the average KL with a small margin
            margin = 0.1  # 10% margin to ensure target is slightly higher than average
            self.target = avg_kl * (1 + margin)
            logger.info(f"LinearKLController: Adaptive target KL update: avg_kl={avg_kl:.4f}, new_target={self.target:.4f}")
        else:
            # During warmup, keep using initial target
            logger.info(f"LinearKLController: Adaptive target KL warmup: step={len(self.kl_history)}/{self.adaptive_window}, target={self.target:.4f}")

class EntropyBudgetController:
    """
    Entropy budget controller for Dual-Game RL algorithm.
    
    Manages the lambda coefficient and entropy budget B according to:
    lambda <- [lambda + alpha_lambda * (B - sum(w_t * H_t))]_+
    B <- B0 * exp(-alpha * step / T)
    
    Adaptive B_t: Uses average of last 10 steps to set per-token budget
    """

    def __init__(self, target, lambda_init=0.0, lambda_lr=0.05, alpha=0.001, total_steps=1000, 
                 adaptive_window=10, adaptive_enabled=True, budget_mix_alpha: float = 0.5):
        """Initialize entropy budget controller.
        
        Args:
            target (float): Initial entropy budget B0 (per-token budget B_t)
            lambda_init (float): Initial lambda coefficient
            lambda_lr (float): Learning rate for lambda updates (alpha_lambda)
            alpha (float): Decay rate parameter for entropy budget
            total_steps (int): Total training steps T
            adaptive_window (int): Window size for adaptive B_t calculation (default: 10)
            adaptive_enabled (bool): Whether to enable adaptive B_t (default: True)
        """
        self.B0 = target
        self.target = target
        self.lambda_init = lambda_init
        self.value = lambda_init
        self.lambda_lr = lambda_lr
        self.alpha = alpha
        self.total_steps = total_steps
        
        # 奖励标准差跟踪
        self.reward_std_history = []
        self.sigma_R_max = None
        self.warmup_steps = min(100, total_steps // 10)  # 前10%步数用于估计σ_{R,max}
        self.min_target_ratio = 0.1  # 最小预算比例
        # 熵预算衰减混合权重 (σ_R 与 使用率)
        self.budget_mix_alpha = max(0.0, min(budget_mix_alpha, 1.0))
        
        # Adaptive B_t parameters
        self.adaptive_enabled = adaptive_enabled
        self.adaptive_window = adaptive_window
        self.wH_history = []  # Store wH_sum / token_count for adaptive calculation
        
        logger.info(f"Using entropy budget controller: value={self.value}, B0={self.B0}, lambda_lr={self.lambda_lr}, alpha={self.alpha}, total_steps={self.total_steps}")
        if self.adaptive_enabled:
            logger.info(f"Adaptive B0 enabled with window size: {self.adaptive_window}")

        
    def update(self, current_wH_sum, token_count):
        """更新lambda系数，使用B_t = B/T的分配方式"""
        if token_count is None or token_count <= 0:
            raise ValueError("token_count is None or <= 0")
            
        # 计算每个token的预算: B_t = B/T
        B_t = self.target / token_count
        logger.info(f"B_t: {B_t}")
        
        # 计算每个token的加权熵
        wH_per_token = current_wH_sum / token_count
        logger.info(f"EntropyBudgetController: wH_per_token: {wH_per_token}")
        
        # 更新lambda: λ <- [λ + α_λ * (wH_per_token - B_t)]_+
        budget_diff = wH_per_token - B_t
        logger.info(f"EntropyBudgetController: budget_diff: {budget_diff}")
        self.value = max(0.0, self.value + self.lambda_lr * budget_diff)
        logger.info(f"EntropyBudgetController: update lambda: {self.value}")
        


    def _update_adaptive_b_0(self, wH_per_token):
        """Update adaptive initial budget B0 based on historical wH_per_token values.
        
        Args:
            wH_per_token (float): Current weighted entropy per token
        """
        # Add current wH_per_token to history
        self.wH_history.append(wH_per_token)
        
        # Keep only the last adaptive_window values
        if len(self.wH_history) > self.adaptive_window:
            self.wH_history = self.wH_history[-self.adaptive_window:]
        
        # Update B0 based on average of last adaptive_window steps
        if len(self.wH_history) >= self.adaptive_window:
            avg_wH_per_token = sum(self.wH_history) / len(self.wH_history)
            # Set B0 to the average wH_per_token with a small margin
            margin = 0.1  # 10% margin to ensure budget is slightly higher than average
            self.B0 = avg_wH_per_token * (1 + margin)
            logger.info(f"EntropyBudgetController: Adaptive B0 update: avg_wH_per_token={avg_wH_per_token:.4f}, new_B0={self.B0:.4f}")
        else:
            # During warmup, keep using initial B0
            logger.info(f"EntropyBudgetController: Adaptive B0 warmup: step={len(self.wH_history)}/{self.adaptive_window}, B0={self.B0:.4f}")



    def decay_target(self, current_step, reward_std=None, usage_ratio: float = None):
        """根据奖励标准差与上一批预算使用率混合调整熵预算"""
        # --------- 统计 σ_R,max ---------
        if reward_std is not None:
            self.reward_std_history.append(reward_std)
            if current_step <= self.warmup_steps:
                self.sigma_R_max = max(self.reward_std_history)

        # 归一化后的奖励方差
        sigma_hat = 0.0
        if reward_std is not None and self.sigma_R_max:
            sigma_hat = reward_std / self.sigma_R_max
            sigma_hat = max(0.0, min(sigma_hat, 1.0))

        # 预算使用率 (已裁剪到 [0,1])
        u_clipped = 1.0
        if usage_ratio is not None and usage_ratio >= 0:
            u_clipped = min(usage_ratio, 1.0)

        # 混合得到衰减因子
        decay_factor = self.budget_mix_alpha * sigma_hat + (1 - self.budget_mix_alpha) * u_clipped
        decay_factor = max(self.min_target_ratio, decay_factor)
        logger.info(f"EntropyBudgetController: decay_factor: {decay_factor}")
        logger.info(f"EntropyBudgetController: B0: {self.B0}")

        self.target = self.B0 * decay_factor
        logger.info(f"EntropyBudgetController: target: {self.target}")


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    elif kl_ctrl.type == "linear":
        # Support both new naming (beta_init/beta_lr) and fallback to kl_coef if provided
        init_beta = getattr(kl_ctrl, "beta_init", getattr(kl_ctrl, "kl_coef", 0.0))
        beta_lr = getattr(kl_ctrl, "beta_lr", 0.01)
        # Extract adaptive parameters with defaults
        adaptive_enabled = getattr(kl_ctrl, "adaptive_enabled", True)
        adaptive_window = getattr(kl_ctrl, "adaptive_window", 10)
        logger.info(f"Using linear KL controller: init_beta={init_beta}, beta_lr={beta_lr}")
        return LinearKLController(
            init_beta=init_beta, 
            target_kl=kl_ctrl.target_kl, 
            beta_lr=beta_lr,
            adaptive_enabled=adaptive_enabled,
            adaptive_window=adaptive_window
        )
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages



def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gpg")
def compute_policy_loss_gpg(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-mean", config=None):
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    pg_losses = -log_prob * advantages

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


@register_policy_loss("dual_game")  # 保持名字不变，配置里无需改
def compute_policy_loss_dual_game(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BNTO 实现：PPO-clip + 重要性加权熵正则 + KL 正则
    - I_t = z(C_proxy) + alpha_I * z(H_proxy) + beta_I * z(D_proxy), clamp>=0
    - 熵梯度用 REINFORCE 形式：lambda * I_t * (-log p - 1)
    - KL 惩罚：beta * (log p - log p0)
    """
    if config is None:
        raise ValueError("config is required for dual_game(BNTO) loss")

    dual_game_config = config.policy_loss.dual_game
    gamma = dual_game_config.get("gamma", 0.8)
    lambda_coef = float(dual_game_config.get("lambda_coef", 0.0))  # 训练器会每步写入
    beta_coef   = float(dual_game_config.get("beta_coef", 0.0))    # 训练器会每步写入

    # BNTO 的 I_t 组合权重（可在 config 里设置）
    alpha_I = float(dual_game_config.get("alpha_I", 0.3))
    beta_I  = float(dual_game_config.get("beta_I", 0.2))
    c_I     = float(dual_game_config.get("c_I", 1.0))

    eps = 1e-6

    # ------- 1) 计算 I_t 的三个分量代理（只用到 log_prob / advantages / mask） -------
    with torch.no_grad():
        # H_proxy：采样动作的惊奇度（与熵强相关）
        surpr = (-log_prob)  # shape (bs, T)
        surpr_mean = verl_F.masked_mean(surpr, response_mask)
        surpr_c = surpr - surpr_mean  # 中心化

        # D_proxy：优势-惊奇协动（中心化）
        D_proxy = advantages * surpr_c  # 保留符号以区分错配方向

        # C_proxy：结构复杂度（时间邻接优势差的强度，近似“局部价值跃迁”）
        adv_shift = torch.roll(advantages, 1, dims=1)
        adv_shift[:, 0] = advantages[:, 0]
        C_proxy = torch.abs(advantages - adv_shift)

        # 批内掩码 z-score 归一化 + 截尾（稳健）
        def _masked_z(x):
            m = verl_F.masked_mean(x, response_mask)
            v = verl_F.masked_mean((x - m) ** 2, response_mask)
            s = torch.sqrt(v + eps)
            z = (x - m) / (s + eps)
            # 3σ 截尾
            return torch.clamp(z, -3.0, 3.0)

        Cn = _masked_z(C_proxy)
        Hn = _masked_z(surpr)
        Dn = _masked_z(D_proxy)

        I_t = c_I * Cn + alpha_I * Hn + beta_I * Dn
        I_t = torch.clamp(I_t, min=0.0)  # 非负，避免反向“奖励”熵
        I_t = I_t * response_mask  # 遮掉无效 token
        # I_t 作为权重信号，不反传
        I_t = I_t.detach()

    # ------- 2) 组装 BNTO 的修正优势 -------
    adv_positive = torch.clamp(advantages, min=0.0)
    adv_negative = torch.clamp(advantages, max=0.0)
    adv_base = adv_positive + gamma * adv_negative

    # 熵项（REINFORCE 形式）：lambda * I_t * (-log p - 1)
    entropy_term = lambda_coef * I_t * (-log_prob - 1.0)

    # KL 项：beta * (log p - log p0)
    kl_term = beta_coef * (log_prob - ref_log_prob)

    adv_modified = adv_base - entropy_term - kl_term

    # ------- 3) PPO-clip 外形保持一致 -------
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -adv_modified * ratio
    cliprange = config.clip_ratio
    clip_low  = config.clip_ratio_low  if config.clip_ratio_low  is not None else cliprange
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    pg_losses2 = -adv_modified * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean((pg_losses2 > pg_losses1).float(), response_mask)

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0, device=pg_loss.device)

# === NEW: Trie-based structural complexity (no critic needed) ==================
import torch

@torch.no_grad()
def compute_structural_complexity_trie(
    responses: torch.Tensor,          # (bs, T) int tokens
    response_mask: torch.Tensor,      # (bs, T) bool or 0/1
    seq_scores: torch.Tensor,         # (bs,)  每条轨迹的“序列分”（如 token_level_rewards.sum(-1))
    group_ids: torch.Tensor,          # (bs,)  同一 prompt 的分组 id (GRPO 的 index)
    reduce: str = "var",              # "var" | "iqr" | "maxmin"
    agg: str = "median",              # 分支分数聚合: "mean" | "median"
    min_branch_size: int = 1,         # 每个分支最小样本数
) -> torch.Tensor:
    """
    基于“前缀树分叉归因”的结构复杂度 C_t：
    对同一组 (同一 prompt) 的 K 条轨迹，在每个位置 t:
      - 只看 response_mask 为 True 的轨迹；
      - 若 t 处出现 >=2 个不同 token，则按 token 分组，对各组汇总该组的“序列分”；
      - 用分支分数的方差/四分位距/IQR/极差来定义该位置的分叉复杂度；
      - 将该复杂度赋给组内所有在 t 位置有效的样本（表示“该前缀状态”的复杂度），无效处为 0。
    无需 critic；与 GRPO/SRPO 的多轨迹天然契合。
    """
    device = responses.device
    bs, T = responses.shape
    C = torch.zeros((bs, T), device=device, dtype=torch.float32)

    # 统一 mask 为 bool
    if response_mask.dtype != torch.bool:
        resp_mask_bool = response_mask > 0
    else:
        resp_mask_bool = response_mask

    # group_ids 统一到 torch.long
    if not torch.is_tensor(group_ids):
        group_ids = torch.as_tensor(group_ids, device=device)
    group_ids = group_ids.to(device=device, dtype=torch.long)

    unique_gids = torch.unique(group_ids)

    for gid in unique_gids:
        idx = (group_ids == gid).nonzero(as_tuple=False).squeeze(-1)  # 该组的样本索引
        if idx.numel() == 0:
            continue

        seqs = responses[idx]                    # (K, T)
        mask = resp_mask_bool[idx]               # (K, T)
        scores = seq_scores[idx].float()         # (K,)

        # 按位置扫描分叉
        for t in range(T):
            valid = mask[:, t]                   # (K,)
            if valid.sum().item() < 2:
                continue

            tok_t = seqs[valid, t]               # 有效样本在 t 的 token
            uniq = torch.unique(tok_t)
            if uniq.numel() < 2:
                continue  # 没有分叉

            # 计算各分支的“分支分数” S(p->v)：分到该 token 的样本的序列分的均值/中位数
            branch_vals = []
            for v in uniq:
                sel = (tok_t == v)               # 有效子集内部的 mask
                if sel.sum().item() < min_branch_size:
                    continue
                # 映射回原 scores：先在 valid 内选择，再用 sel 二次筛
                s = scores[valid][sel]
                if s.numel() == 0:
                    continue
                if agg == "mean":
                    branch_vals.append(s.mean())
                else:  # "median"
                    branch_vals.append(s.median())

            if len(branch_vals) < 2:
                continue  # 有效分支不足两类

            branch_vals = torch.stack(branch_vals)  # (B,)

            # 分支差异 → 结构复杂度
            if reduce == "var":
                c_branch = torch.var(branch_vals, unbiased=False)
            elif reduce == "iqr":
                q = torch.quantile(branch_vals, torch.tensor([0.25, 0.75], device=device))
                c_branch = q[1] - q[0]
            elif reduce == "maxmin":
                c_branch = branch_vals.max() - branch_vals.min()
            else:
                raise ValueError(f"Unknown reduce={reduce}")

            # 将该位置的复杂度赋给“该组在 t 位置的所有有效样本”
            # 解释：C_t 是“该前缀状态”的属性，与该时刻选了哪个 token 无关
            C[idx[valid], t] = c_branch

    return C
