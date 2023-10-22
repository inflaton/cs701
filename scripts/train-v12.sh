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

EPOCHS=30
BATCH=64

echo Running python train-all-phases-k-fold.py --epochs=$EPOCHS --batch=$BATCH --model=2
python train-all-phases-k-fold.py --epochs=$EPOCHS --batch=$BATCH --model=2

./validate.sh
