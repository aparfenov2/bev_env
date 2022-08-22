set -ex
[ "$1" == "--install-poetry" ] && {
    mkdir .app
}

[ "$1" == "--inner" ] && {
    shift
    PATH="/app/.local/bin:$PATH"

    [ "$1" == "--install-poetry" ] && {
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
        exit 0
    }
    . .venv/bin/activate || true
    export PYTHONDONTWRITEBYTECODE=1
    $@
    exit $?
}

image="ml-py38-gpu"

docker build -t $image -f Dockerfile /var/mail

VOLUMES=()
for f in $(find . -type l); do
    [ -e "$f" ] && {
        VOLUMES+=("-v $(readlink -f $f):/cdir/$f")
    }
done

docker run -ti --rm \
    --gpus all \
    --network host \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e XAUTHORITY \
    -v $PWD:/cdir \
    -v $(readlink -f .app):/app \
    ${VOLUMES[@]} \
    -w /cdir \
    $image bash /cdir/$0 --inner $@
