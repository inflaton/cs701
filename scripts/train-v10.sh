#!/bin/sh
BASEDIR=$(dirname "$0")
cd $BASEDIR
echo Current Directory:
pwd

nvidia-smi
uname -a
cat /etc/os-release
lscpu
grep MemTotal /proc/meminfo

EPOCHS=20
BATCH=32

echo Running python train-all-phases-k-fold.py --epochs=$EPOCHS --batch=$BATCH
python train-all-phases-k-fold.py --epochs=$EPOCHS --batch=$BATCH

./validate.sh
