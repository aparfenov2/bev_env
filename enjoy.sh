bash run.sh python enjoy.py --env BEVEnv-v1 --gym-packages bev_env --folder logs \
    --env-kwargs const_dt:0.1 random_pos:True \
    obstacle_done:True max_episode_steps:500 init_logging:True
