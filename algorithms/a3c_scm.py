# create by FA

"""Note: Keep in sync with changes to VTraceTFPolicy."""

from __future__ import absolute_import, division, print_function

from ray.rllib.agents.a3c.a3c import make_async_optimizer, validate_config
from ray.rllib.agents.a3c.a3c_tf_policy import postprocess_advantages
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf

from algorithms.common_funcs_moa import (
    get_moa_mixins,
    moa_postprocess_trajectory,
    validate_moa_config,
)
from algorithms.common_funcs_scm import (
    SOCIAL_CURIOSITY_REWARD,
    SCMResetConfigMixin,
    get_curiosity_mixins,
    scm_fetches,
    scm_postprocess_trajectory,
    setup_scm_loss,
    setup_scm_mixins,
    validate_scm_config,
)
from algorithms.a3c_moa import (
    actor_critic_loss,
    stats,
    grad_stats,
    add_value_function_fetch,
    clip_gradients,
    setup_mixins,
    ValueNetworkMixin,
)

tf = try_import_tf()


def postprocess_a3c_scm(policy, sample_batch, other_agent_batches=None, episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    batch = moa_postprocess_trajectory(policy, sample_batch)
    batch = scm_postprocess_trajectory(policy, batch, other_agent_batches)  # FA: I added other_agent_batches to input list.
    batch = postprocess_advantages(policy, batch)
    return batch


def loss_with_scm(policy, model, dist_class, train_batch):
    """
    Calculate PPO loss with SCM and MOA loss
    :return: Combined A3C+MOA+SCM loss
    """
    _ = actor_critic_loss(policy, model, dist_class, train_batch)

    scm_loss = setup_scm_loss(policy, train_batch)
    policy.scm_loss = scm_loss.total_loss

    # policy.loss.total_loss has already been instantiated in actor_critic_loss
    policy.loss.total_loss += scm_loss.total_loss
    return policy.loss.total_loss


def extra_scm_fetches(policy):
    """
    Adds value function, logits, moa predictions, SCM loss/reward to experience train_batches.
    :return: Updated fetches
    """
    a3c_fetches = add_value_function_fetch(policy)
    a3c_fetches.update(scm_fetches(policy))
    return a3c_fetches


def extra_scm_stats(policy, train_batch):
    """
    Add stats that are logged in progress.csv
    :return: Combined A3C+MOA+SCM stats
    """
    scm_stats = stats(policy, train_batch)
    scm_stats = {
        **scm_stats,
        "cur_curiosity_reward_weight": tf.cast(
            policy.cur_curiosity_reward_weight_tensor, tf.float32
        ),
        SOCIAL_CURIOSITY_REWARD: train_batch[SOCIAL_CURIOSITY_REWARD],
        "inequity_aversion_reward": train_batch["inequity_aversion_reward"],
        "extrinsic_reward": train_batch["extrinsic_reward"],
        "scm_loss": policy.scm_loss,
    }
    return scm_stats


def mixins(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    setup_scm_mixins(policy, obs_space, action_space, config)


def validate_a3c_scm_config(config):
    """
    Validates the A3C+MOA+SCM config
    :param config: The config to validate
    """
    validate_scm_config(config)
    validate_moa_config(config)
    validate_config(config)


def build_a3c_scm_trainer(scm_config):
    tf.keras.backend.set_floatx("float32")
    trainer_name = "SCMA3CTrainer"
    scm_config["use_gae"] = False

    scm_a3c_policy = build_tf_policy(
        name="A3CAuxTFPolicy",
        get_default_config=lambda: scm_config,
        loss_fn=loss_with_scm,
        stats_fn=extra_scm_stats,
        grad_stats_fn=grad_stats,
        gradients_fn=clip_gradients,
        postprocess_fn=postprocess_a3c_scm,
        extra_action_fetches_fn=extra_scm_fetches,
        before_loss_init=mixins,
        mixins=[ValueNetworkMixin, LearningRateSchedule]
        + get_moa_mixins()
        + get_curiosity_mixins(),
    )

    scm_a3c_trainer = build_trainer(
        name=trainer_name,
        default_policy=scm_a3c_policy,
        default_config=scm_config,
        validate_config=validate_a3c_scm_config,
        mixins=[SCMResetConfigMixin],
        make_policy_optimizer=make_async_optimizer,
    )

    return scm_a3c_trainer
