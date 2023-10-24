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

EPOCHS=20
BATCH=128
MODEL=1

echo Running python train-all-knowledge-distillation.py --epochs=$EPOCHS --batch=$BATCH --model=$MODEL
python train-all-knowledge-distillation.py --epochs=$EPOCHS --batch=$BATCH --model=$MODEL

a=1
until [ $a -gt 10 ]
do
    echo Running python validate.py --batch=$BATCH --phase=$a --model=$MODEL
    python validate.py --batch=$BATCH --phase=$a --model=$MODEL
     
    # increment the value
    a=`expr $a + 1`
done
