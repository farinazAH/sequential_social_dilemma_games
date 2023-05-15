# FA: begin
import numpy as np

base_inequityaversion = 2  # 1 => baseline  ,  2 => IA

# FA: end

class BaselineResetConfigMixin(object):
    @staticmethod
    def reset_policies(policies, new_config):
        for policy in policies:
            policy.entropy_coeff_schedule.value = lambda _: new_config["entropy_coeff"]
            policy.config["entropy_coeff"] = new_config["entropy_coeff"]
            policy.lr_schedule.value = lambda _: new_config["lr"]
            policy.config["lr"] = new_config["lr"]

    def reset_config(self, new_config):
        self.reset_policies(self.optimizer.policies.values(), new_config)
        self.config = new_config
        return True

# FA: begin

# def iv_fetches(policy):
#     """Adds inequity aversion e_j e_i to experience train_batches."""
#     return {
#         'reward_eligibility': policy.model.reward_eligibility(),
#     }


def baseline_inequity_aversion_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
    # Weigh inequity_aversion reward and add to batch.
    if base_inequityaversion_empathy == 2:
        intrinsic_reward = inequity_aversion_eligibility(sample_batch, other_agent_batches)
        # Add to trajectory
        sample_batch["inequity_aversion_reward"] = intrinsic_reward
        sample_batch["extrinsic_reward"] = sample_batch["rewards"]
        sample_batch["rewards"] = sample_batch["rewards"] + intrinsic_reward    
    return sample_batch


def inequity_aversion_eligibility(sample_batch, other_agent_batches):
    my_reward_eligibility = np.array([0.0] * len(sample_batch['rewards']))
    my_reward_eligibility[0] = sample_batch['rewards'][0]
    for i in range(1, len(my_reward_eligibility)):
        my_reward_eligibility[i] = 0.9 * 0.9 * my_reward_eligibility[i-1] + sample_batch['rewards'][i]
    # print('*************')
    # print(other_agent_batches)
    # print([(a, b) for a, b in zip(sample_batch['rewards'], my_reward_eligibility)])
    # print('+++++++++++++++++++')
    inequity_reward = np.array([0.0] * len(sample_batch['rewards']))
    if other_agent_batches is not None:
        alpha_i = 0 #5.0
        beta_i = 0.05
        N_1 = len(other_agent_batches)
        alpha = -1 * (alpha_i / N_1)
        beta = -1 * (beta_i / N_1)
        disadvantageous_inequity = np.array([0.0] * len(sample_batch['rewards']))
        advantageous_inequity = np.array([0.0] * len(sample_batch['rewards']))
        for key, value in other_agent_batches.items():
            other_eligibility_rewards = np.array([0.0] * len(sample_batch['rewards']))
            other_eligibility_rewards[0] = value[1]['rewards'][0]
            for i in range(1, len(other_eligibility_rewards)):
                other_eligibility_rewards[i] = 0.9 * 0.9 * other_eligibility_rewards[i - 1] + value[1]['rewards'][i]
            disadvantageous_inequity += np.maximum(other_eligibility_rewards-my_reward_eligibility, 0.0)
            advantageous_inequity += np.maximum(my_reward_eligibility-other_eligibility_rewards, 0.0)
        inequity_reward = alpha * disadvantageous_inequity + beta * advantageous_inequity
    return inequity_reward

# FA: end
