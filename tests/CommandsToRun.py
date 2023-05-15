# tensorboard --logdir ~/ray_results


# rllib rollout ~/ray_results/cleanup_baseline_PPO/BaselinePPOTrainer_cleanup_env_0_2021-03-17_22-42-52brls4y1h/checkpoint_300/checkpoint-300 --run PPO --env cleanup --steps 1000

# python3 train.py --model moa  --num_agents 5 --stop_at_timesteps_total 5e8

# python train.py --env cleanup --model baseline --algorithm A3C --num_agents 5 --num_workers 6 --rollout_fragment_length 1000 --num_envs_per_worker 16 --stop_at_timesteps_total $((500 * 10 ** 6)) --memory $((160 * 10 ** 9)) --cpus_per_worker 1 --gpus_per_worker 0 --gpus_for_driver 1 --cpus_for_driver 0 --num_samples 5 --entropy_coeff 0.00176 --lr_schedule_steps 0 20000000 --lr_schedule_weights .00126 .000012


