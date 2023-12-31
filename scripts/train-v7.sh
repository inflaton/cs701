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

# EPOCHS=50 -v6
EPOCHS=100
BATCH=32

# v6
# echo Running python train.py --epochs=$EPOCHS --batch=$BATCH --phase=1 --iteration=1 --folder=data/v3-resnext101_32x8d/checkpoints_phase_10/ --checkpoint=22
# python train.py --epochs=$EPOCHS --batch=$BATCH --phase=1 --iteration=1 --folder=data/v3-resnext101_32x8d/checkpoints_phase_10/ --checkpoint=22

# v7
echo Running python train.py --epochs=$EPOCHS --batch=$BATCH --phase=1 --iteration=2 --folder=data/checkpoints_phase_10/ --checkpoint=39
python train.py --epochs=$EPOCHS --batch=$BATCH --phase=1 --iteration=2 --folder=data/checkpoints_phase_10/ --checkpoint=39

a=2
until [ $a -gt 10 ]
do
    echo Running python train.py --epochs=$EPOCHS --batch=$BATCH --phase=$a
    python train.py --epochs=$EPOCHS --batch=$BATCH --phase=$a
     
    # increment the value
    a=`expr $a + 1`
done

./validate.sh
