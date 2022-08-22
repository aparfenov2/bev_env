bash run.sh python train.py --env BEVEnv-v1 --gym-packages bev_env \
    --env-kwargs twist_only:True const_dt:0.1 random_pos:True \
    obstacle_done:True max_episode_steps:500 init_logging:True
