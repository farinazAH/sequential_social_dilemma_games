from __future__ import absolute_import, division, print_function

from ray.rllib.agents.a3c.a3c import make_async_optimizer, validate_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy import build_tf_policy
from ray.rllib.agents.a3c.a3c_tf_policy import (
    ValueNetworkMixin,
    clip_gradients,
    postprocess_advantages,
    setup_mixins,
    LearningRateSchedule,
    add_value_function_fetch,
    grad_stats,
    actor_critic_loss,
    stats,
)
from algorithms.common_funcs_baseline import (
    baseline_inequity_aversion_postprocess_trajectory,  # FA
    base_inequityaversion,  # FA
)


def extra_iv_stats(policy, train_batch):  # FA
    """
    Add stats that are logged in progress.csv
    :return: Combined a3c+inequityAversion_empathy stats
    """
    base_stats = stats(policy, train_batch)
    if base_inequityaversion == 2:
        base_stats = {
            **base_stats,
            "inequity_aversion_reward": train_batch["inequity_aversion_reward"],
            "extrinsic_reward": train_batch["extrinsic_reward"],
        }
    return base_stats


def postprocess_a3c_baseline(policy, sample_batch, other_agent_batches=None, episode=None):  # FA
    """
    Add the inequity_aversion reward to the trajectory.
    Then, add the policy logits, VF preds, and advantages to the trajectory.
    :return: Updated trajectory (batch)
    """

    batch = baseline_inequity_aversion_postprocess_trajectory(policy, sample_batch, other_agent_batches)
    batch = postprocess_advantages(policy, batch)
    # print(batch["inequity_aversion_reward"])

    return batch


def build_a3c_baseline_trainer(config):

    policy = build_tf_policy(
        name="A3CTFPolicy",
        get_default_config=lambda: config,
        loss_fn=actor_critic_loss,
        stats_fn=extra_iv_stats,  # FA
        grad_stats_fn=grad_stats,
        gradients_fn=clip_gradients,
        postprocess_fn=postprocess_a3c_baseline,  # FA
        extra_action_fetches_fn=add_value_function_fetch,
        before_loss_init=setup_mixins,
        mixins=[ValueNetworkMixin, LearningRateSchedule]
    )

    a3c_trainer = build_trainer(
        name="BaselineA3CTrainer",
        default_config=config,
        default_policy=policy,   # FA
        validate_config=validate_config,
        make_policy_optimizer=make_async_optimizer,
    )
    return a3c_trainer
