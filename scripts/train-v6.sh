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

EPOCHS=50
BATCH=32

echo Running python train.py --epochs=$EPOCHS --batch=$BATCH --phase=1 --iteration=1 --folder=data/v3-resnext101_32x8d/checkpoints_phase_10/ --checkpoint=22
Running python train.py --epochs=$EPOCHS --batch=$BATCH --phase=1 --iteration=1 --folder=data/v3-resnext101_32x8d/checkpoints_phase_10/ --checkpoint=22

a=2
until [ $a -gt 10 ]
do
    echo Running python train.py --epochs=$EPOCHS --batch=$BATCH --phase=$a
    python train.py --epochs=$EPOCHS --batch=$BATCH --phase=$a
     
    # increment the value
    a=`expr $a + 1`
done
