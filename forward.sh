#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "specify a port."
    return 1
fi

ssh -K -N -L $1:localhost:$1 bh511@dev-gpu-bh511.cl.cam.ac.uk
