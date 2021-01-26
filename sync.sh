#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "usage: sync.sh <file or folder> <direction>"
    return 1
fi

if [ "$2" = "up" ]; then
    rsync -av -e 'ssh -K' --progress --exclude-from=.rsyncignore $1 bh511@dev-gpu-bh511.cl.cam.ac.uk:~/$1
elif [ "$2" = "down" ]; then
    rsync -av -e 'ssh -K' --progress --exclude-from=.rsyncignore bh511@dev-gpu-bh511.cl.cam.ac.uk:~/$1 $1
else
    echo "invalid direction"
fi
