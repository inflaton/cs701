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

BATCH=32
EPOCHS=50

# python train.py --epochs=20 --batch=$BATCH --phase=1
# python train.py --epochs=20 --batch=$BATCH --phase=2 --checkpoint=7
# python train.py --epochs=30 --batch=$BATCH --phase=3 --checkpoint=15
# python train.py --epochs=40 --batch=$BATCH --phase=4 --checkpoint=21
# python train.py --epochs=40 --batch=$BATCH --phase=5 --checkpoint=16
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=6 --checkpoint=17
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=7 --checkpoint=22
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=8 --checkpoint=14
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=9 --checkpoint=34
python train.py --epochs=$EPOCHS --batch=$BATCH --phase=10 --checkpoint=19

