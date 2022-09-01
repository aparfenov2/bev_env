
EXPERIMENT_NAME="gym-train"
# bash run_local.sh --host black_over_pi --user usr -e ${EXPERIMENT_NAME} --docker --src svn/ --train
# bash run_local.sh --host black_over_pi --user usr -e ${EXPERIMENT_NAME} --docker --daemon --tg --src svn/ --train
bash run_local.sh --host local --user usr -e ${EXPERIMENT_NAME} --docker --src svn/ --train
# bash run_local.sh --host black_over_pi --user usr -e ${EXPERIMENT_NAME} --docker --src svn/ --train
