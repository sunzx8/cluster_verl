# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import ray
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import simple_timer

from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager, WorkerType
from typing import Optional


class RayEntropyTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, processor, reward_fn, val_reward_fn, train_dataset, val_dataset, collate_fn, train_sampler, device_name)
        
        # Initialize entropy budget controller for dual-game RL after total_training_steps is calculated
        if hasattr(self.config, 'entropy_budget'):
            from verl.trainer.ppo.core_algos import EntropyBudgetController
            # Extract parameters with proper defaults
            target = self.config.entropy_budget.get('target', 5.0)
            lambda_init = self.config.entropy_budget.get('lambda_init', 0.0)
            lambda_lr = self.config.entropy_budget.get('lambda_lr', 0.05)
            alpha = self.config.entropy_budget.get('alpha', 0.001)
            max_lambda = self.config.entropy_budget.get('max_lambda', 10.0)
            wH_clip_range = self.config.entropy_budget.get('wH_clip_range', [0.0, 10.0])
            # Use actual total training steps
            total_steps = self.total_training_steps
            
            # Extract adaptive parameters with defaults
            adaptive_enabled = self.config.entropy_budget.get('adaptive_enabled', True)
            adaptive_window = self.config.entropy_budget.get('adaptive_window', 10)
            
            self.entropy_ctrl = EntropyBudgetController(
                target=target,
                lambda_init=lambda_init,
                lambda_lr=lambda_lr,
                alpha=alpha,
                total_steps=total_steps,
                adaptive_enabled=adaptive_enabled,
                adaptive_window=adaptive_window
            )
        else:
            self.entropy_ctrl = None

        # Initialize beta (KL) controller for dual-game RL, independent of reward-KL path
        if hasattr(self.config, 'critic'):
            from verl.trainer.ppo.core_algos import get_kl_controller
            self.beta_ctrl = get_kl_controller(self.config.critic.kl_ctrl)
        else:
            self.beta_ctrl = None

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Update dual-game coefficients before actor update
                        if (self.entropy_ctrl is not None and 
                            hasattr(self.config.actor_rollout_ref.actor.policy_loss, 'loss_mode') and
                            self.config.actor_rollout_ref.actor.policy_loss.loss_mode == "dual_game"):
                            
                           with marked_timer("dual_game_update", timing_raw, color="purple"):
                                # === 当前策略 logp/entropy、adv/mask ===
                                curr = self.actor_rollout_wg.compute_log_prob(batch)
                                logp = curr.batch["old_log_probs"]               # (bs, T)
                                H    = curr.batch["entropys"]                    # (bs, T)
                                resp_mask = batch.batch["response_mask"].to(logp.dtype)
                                adv  = batch.batch["advantages"]                 # (bs, T)

                                # 取 responses / seq_scores / group_ids
                                responses = batch.batch["responses"]             # (bs, T) int tokens
                                response_mask_bool = batch.batch["response_mask"].bool()  # (bs, T)
                                seq_scores = batch.batch["token_level_rewards"].sum(dim=-1)  # (bs,)

                                # group_ids 兼容性读取
                                if "index" in batch.non_tensor_batch:
                                    group_ids = batch.non_tensor_batch["index"]
                                elif "index" in batch.batch:
                                    group_ids = batch.batch["index"]
                                else:
                                    raise KeyError("BNTO(trie): group index 'index' not found in batch (non_tensor_batch or batch).")

                                # === 计算 Trie 分叉归因（top-k）的 C_proxy（无需 critic） ===
                                C_proxy = core_algos.compute_structural_complexity_trie(
                                    responses=responses,
                                    response_mask=response_mask_bool,
                                    seq_scores=seq_scores,
                                    group_ids=group_ids,
                                    reduce=getattr(self.config.actor_rollout_ref.actor.policy_loss.dual_game, "c_reduce", "var"),
                                    agg=getattr(self.config.actor_rollout_ref.actor.policy_loss.dual_game, "c_agg", "median"),
                                    min_branch_size=getattr(self.config.actor_rollout_ref.actor.policy_loss.dual_game, "c_min_branch_size", 1),
                                ).to(logp.dtype)

                                # === 差异度与中心化（H 的单样本近似；后面算 D）===
                                eps = 1e-6
                                surpr = (-logp)
                                # 每条样本内做加权平均（掩码）
                                surpr_mean = torch.sum(surpr * resp_mask, dim=1, keepdim=True) / (torch.sum(resp_mask, dim=1, keepdim=True) + eps)
                                surpr_c = surpr - surpr_mean
                                D_proxy = adv * surpr_c

                                # === 掩码内 z-score 归一化 + 截尾 ===
                                def masked_z(x, m):
                                    num = torch.sum(x * m, dim=1, keepdim=True)
                                    den = torch.sum(m, dim=1, keepdim=True) + eps
                                    mean = num / den
                                    var  = torch.sum(((x - mean) ** 2) * m, dim=1, keepdim=True) / den
                                    std  = torch.sqrt(var + eps)
                                    z = (x - mean) / (std + eps)
                                    return torch.clamp(z, -3.0, 3.0)

                                alpha_I = float(getattr(self.config.actor_rollout_ref.actor.policy_loss.dual_game, "alpha_I", 0.3))
                                beta_I  = float(getattr(self.config.actor_rollout_ref.actor.policy_loss.dual_game, "beta_I", 0.2))
                                c_I     = float(getattr(self.config.actor_rollout_ref.actor.policy_loss.dual_game, "c_I", 1.0))

                                Cn = masked_z(C_proxy, resp_mask)
                                Hn = masked_z(surpr,   resp_mask)
                                Dn = masked_z(D_proxy, resp_mask)

                                I_t = torch.clamp(c_I * Cn + alpha_I * Hn + beta_I * Dn, min=0.0) * resp_mask
                                I_t = I_t.detach()

                                # === wH 累计 & 控制器更新（与原来一致，只是 w 换成 I_t）===
                                wH_sum = torch.sum(I_t * H * resp_mask)
                                token_count = torch.sum(resp_mask)

                                self.entropy_ctrl.update(current_wH_sum=wH_sum.item(), token_count=int(token_count.item()))

                                token_rewards = batch.batch["token_level_rewards"]
                                valid_rewards = token_rewards[response_mask_bool]
                                reward_std = torch.std(valid_rewards).item() if valid_rewards.numel() > 0 else 0.0
                                usage_ratio = (wH_sum.item() / (self.entropy_ctrl.target + 1e-8))
                                self.entropy_ctrl.decay_target(self.global_steps, reward_std=reward_std, usage_ratio=usage_ratio)

                                # === KL 控制器（β）更新 & 广播===
                                beta_val = 0.0
                                kld_value = 0.0
                                if self.beta_ctrl is not None and "ref_log_probs" in batch.batch:
                                    kld_mat = core_algos.kl_penalty(
                                        logprob=batch.batch["old_log_probs"],
                                        ref_logprob=batch.batch["ref_log_probs"],
                                        kl_penalty="low_var_kl",
                                    )
                                    kld_value = torch.sum(kld_mat * resp_mask) / (token_count + 1e-8)
                                    self.beta_ctrl.update(current_kl=float(kld_value.item()), n_steps=len(batch))
                                    beta_val = float(self.beta_ctrl.value)

                                # 写回 config 并广播
                                if not hasattr(self.config.actor_rollout_ref.actor.policy_loss, 'dual_game'):
                                    from omegaconf import DictConfig
                                    with open_dict(self.config.actor_rollout_ref.actor.policy_loss):
                                        self.config.actor_rollout_ref.actor.policy_loss.dual_game = DictConfig({})
                                self.config.actor_rollout_ref.actor.policy_loss.dual_game.lambda_coef = float(self.entropy_ctrl.value)
                                self.config.actor_rollout_ref.actor.policy_loss.dual_game.beta_coef   = float(beta_val)
                                self.config.actor_rollout_ref.actor.policy_loss.dual_game.alpha_I     = float(alpha_I)
                                self.config.actor_rollout_ref.actor.policy_loss.dual_game.beta_I      = float(beta_I)
                                self.config.actor_rollout_ref.actor.policy_loss.dual_game.c_I         = float(c_I)

                                self.actor_rollout_wg.execute_all_sync(
                                    "set_loss_coefficients",
                                    lambda_coef=float(self.entropy_ctrl.value),
                                    kl_coef=float(beta_val),
                                )

                                # 记录指标
                                B_token = float(self.entropy_ctrl.target) / (float(token_count.item()) + 1e-8)
                                token_entropy_mean = (torch.sum(H * resp_mask) / (token_count + 1e-8)).item()
                                metrics.update({
                                    "dual_game/per_token_budget_Bt": B_token,
                                    "dual_game/wH_per_token": (wH_sum.item() / (token_count.item() + 1e-8)),
                                    "dual_game/lambda": float(self.entropy_ctrl.value),
                                    "dual_game/beta": beta_val,
                                    "dual_game/KL_per_token": float(kld_value),
                                    "dual_game/token_entropy_mean": token_entropy_mean,
                                    "dual_game/reward_std": reward_std,
                                    "dual_game/token_count": int(token_count.item()),
                                })


                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    def detect_entropy_collapse(self, wH_per_token, step):
        """检测熵崩塌并自动调整"""
        if step > 2:  # 从第3步开始检测
            # 如果wH下降超过90%，认为发生熵崩塌
            if wH_per_token < 0.0001:  # 阈值可调
                print(f" ENTROPY COLLAPSE DETECTED! wH_per_token={wH_per_token:.8f}")
                
                # 自动调整λ和预算
                self.entropy_ctrl.value *= 0.1  # λ减少90%
                self.entropy_ctrl.target = max(0.001, wH_per_token * 2)  # 预算设为当前消耗的2倍
                
                print(f"🔧 Auto-adjustment: λ={self.entropy_ctrl.value:.6f}, B_t={self.entropy_ctrl.target:.6f}")
                return True
        return False
