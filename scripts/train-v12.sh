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

a=1
until [ $a -gt 10 ]
do
    echo Running python validate.py --batch=$BATCH --phase=$a --model=2
    python validate.py --batch=$BATCH --phase=$a --model=2
     
    # increment the value
    a=`expr $a + 1`
done
