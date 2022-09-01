Среда gym для обучения обьезду препятсвий

![static_snapshot](docs/bev_env.png)

# requirements
    - NVIDIA card

# install
git clone https://github.com/kantengri/bev_env
# install poetry
bash run.sh --install-poetry
# install environment
bash run.sh poetry install

# run
# Ручное управление
bash run.sh python manual_control.py

Управление:
! основное окно должно быть в фокусе
управление стрелками:
    LEFT/RIGHT - поворот влево-вправо
    UP/DOWN - движение вперед/назад
PGUP/PGDOWN - увеличить/уменьшить шаг

# training
bash run.sh python train.py

# training on remote/local host
to use remote deployment script the directory structure should be like this:
    svn          # this repo
    run_local.sh # symlink from svn/run_local.sh
    train-remotely.sh     # symlink from svn/train-remotely.sh
To start training run train-remotely.sh. Edit it to specify host ("local" or "<host_name>"), experiment name and jenkins_entry.sh prgs. After deployment to a separate host/folder jenkins_entry.sh is called with specified args.

To run locally you need first to
1. build docker image with run.sh --build
2. create .app folder         # this will be mapped to virtual home directory
3. run.sh --install-poetry    # install poetry into .app
4. run.sh poetry init         # poetry will create virtual environment in .venv and install all dependencies from project.toml

The same is on remote host,
1. run train-remotely.sh once. It will end with error, but create working directory on remote host under ~/jenkins_experiments. ssh & Cd into that directory and setup environment as you did on localhost. Basically, you need symlinks to .app & .venv folders in working directory. You may tar them locally & upload on remote host, then just symlink it. (Remove symlinks from local folder first, they wil be tar-ed (~3Gb) with sources otherwise. Yo do not want to upload 3Gb each new run.)

After environament set up, you can start exploring some python scripts. For example, Try bash train.sh or bash run.sh python manual_control.py