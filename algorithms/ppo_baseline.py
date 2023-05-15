from __future__ import absolute_import, division, print_function

from ray.rllib.agents.ppo.ppo import (
    choose_policy_optimizer,
    update_kl,
    validate_config,
    warn_about_bad_reward_scales,
)
from ray.rllib.agents.ppo.ppo_tf_policy import (
    KLCoeffMixin,
    ValueNetworkMixin,
    clip_gradients,
    kl_and_loss_stats,
    postprocess_ppo_gae,
    ppo_surrogate_loss,
    setup_config,
    setup_mixins,
    vf_preds_fetches,
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy import build_tf_policy
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule

from algorithms.common_funcs_baseline import (
    BaselineResetConfigMixin,
    # iv_fetches,  # FA
    baseline_inequity_aversion_postprocess_trajectory,  # FA
    base_inequityaversion  # FA
)


def extra_iv_fetches(policy):  # FA
    """
    Adds value function, logits, moa predictions to experience train_batches.
    :return: Updated fetches
    """
    ppo_fetches = vf_preds_fetches(policy)
    # ppo_fetches.update(iv_fetches(policy))
    return ppo_fetches


def extra_iv_stats(policy, train_batch):  # FA
    """
    Add stats that are logged in progress.csv
    :return: Combined PPO+inequityAversion_empathy stats
    """
    base_stats = kl_and_loss_stats(policy, train_batch)
    if base_inequityaversion == 2:
        base_stats = {
            **base_stats,
            "inequity_aversion_reward": train_batch["inequity_aversion_reward"],
            "extrinsic_reward": train_batch["extrinsic_reward"],
        }    
    return base_stats


def postprocess_ppo_baseline(policy, sample_batch, other_agent_batches=None, episode=None):  # FA
    """
    Add the inequity_aversion reward to the trajectory.
    Then, add the policy logits, VF preds, and advantages to the trajectory.
    :return: Updated trajectory (batch)
    """
    batch = baseline_inequity_aversion_postprocess_trajectory(policy, sample_batch, other_agent_batches)

    batch = postprocess_ppo_gae(policy, batch)
    return batch


def build_ppo_baseline_trainer(config):
    """
    Creates a PPO policy class, then creates a trainer with this policy.
    :param config: The configuration dictionary.
    :return: A new PPO trainer.
    """
    policy = build_tf_policy(
        name="PPOTFPolicy",
        get_default_config=lambda: config,
        loss_fn=ppo_surrogate_loss,
        stats_fn=extra_iv_stats,  # FA
        extra_action_fetches_fn=extra_iv_fetches,  # FA
        postprocess_fn=postprocess_ppo_baseline,  # FA
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin],
    )

    ppo_trainer = build_trainer(
        name="BaselinePPOTrainer",
        make_policy_optimizer=choose_policy_optimizer,
        default_policy=policy,
        default_config=config,
        validate_config=validate_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
        mixins=[BaselineResetConfigMixin],
    )
    return ppo_trainer
