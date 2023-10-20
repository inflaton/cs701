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

EPOCHS=100
BATCH=32

a=1
until [ $a -gt 10 ]
do
    echo Running python train.py --epochs=$EPOCHS --batch=$BATCH --phase=$a
    python train.py --epochs=$EPOCHS --batch=$BATCH --phase=$a
     
    # increment the value
    a=`expr $a + 1`
done
