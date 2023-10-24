#!/bin/sh
BASEDIR=$(dirname "$0")
cd $BASEDIR
echo Current Directory:
pwd

nvidia-smi
# uname -a
# cat /etc/os-release
# lscpu
# grep MemTotal /proc/meminfo

EPOCHS=50
BATCH=16

echo Running python train-xu-optimized.py --epochs=$EPOCHS --batch=$BATCH
python train-xu-optimized.py --epochs=$EPOCHS --batch=$BATCH
