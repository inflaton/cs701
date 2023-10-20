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

python validate.py --checkpoint=7 --batch=$BATCH --phase=1
python validate.py --checkpoint=15 --batch=$BATCH --phase=2
python validate.py --checkpoint=21 --batch=$BATCH --phase=3
python validate.py --checkpoint=16 --batch=$BATCH --phase=4
python validate.py --checkpoint=17 --batch=$BATCH --phase=5
python validate.py --checkpoint=22 --batch=$BATCH --phase=6
python validate.py --checkpoint=14 --batch=$BATCH --phase=7
python validate.py --checkpoint=34 --batch=$BATCH --phase=8
python validate.py --checkpoint=19 --batch=$BATCH --phase=9
python validate.py --checkpoint=12 --batch=$BATCH --phase=10

