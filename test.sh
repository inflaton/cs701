#!/bin/sh
BASEDIR=$(dirname "$0")
cd $BASEDIR
echo Current Directory:
pwd

BATCH=16
MODEL=2

a=1
until [ $a -gt 10 ]
do
    echo Running python validate.py --val_or_test=-1 --batch=$BATCH --phase=$a --model=$MODEL
    python validate.py --val_or_test=-1 --batch=$BATCH --phase=$a --model=$MODEL
     
    # increment the value
    a=`expr $a + 1`
done
