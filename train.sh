bash run.sh python train.py --algo dqn --env BEVEnv-discrete-v1 --gym-packages bev_env \
    --env-kwargs const_dt:0.1 random_pos:True \
    obstacle_done:True max_episode_steps:500 init_logging:True render_in_step:True
exit 0
bash run.sh python train.py --algo ppo --env BEVEnv-v1 --gym-packages bev_env \
    --env-kwargs twist_only:True const_dt:0.1 random_pos:True \
    obstacle_done:True max_episode_steps:500 init_logging:True
