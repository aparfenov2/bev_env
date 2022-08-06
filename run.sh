set -ex
[ "$1" == "--inner" ] && {
    shift
    PATH="/app/.local/bin:$PATH"
    # curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
    . .venv/bin/activate
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
