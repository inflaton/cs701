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

EPOCHS=10
#EPOCHS=100
BATCH=32

a=1
# -gt is greater than operator
 
#Iterate the loop until a is greater than 10
# until [ $a -gt 10 ]
until [ $a -gt 1 ]
do
    echo Running python train.py --epochs=$EPOCHS --batch=$BATCH --phase=$a
     
    # increment the value
    a=`expr $a + 1`
done
