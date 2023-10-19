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

# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=1
python train.py --epochs=$EPOCHS --batch=$BATCH --phase=2 --checkpoint=7
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=3
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=4
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=5
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=6
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=7
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=8
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=9
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=10

