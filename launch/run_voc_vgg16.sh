#!/bin/bash

## Your values here:
#
DS=pascal_voc
EXP=
RUN_ID=
#
##

#
# Script
#

LOG_DIR=logs/${DS}/${EXP}
CMD="python train.py --dataset $DS --cfg configs/voc_vgg16.yaml --exp $EXP --run $RUN_ID"
LOG_FILE=$LOG_DIR/${RUN_ID}.log

if [ ! -d "$LOG_DIR" ]; then
  echo "Creating directory $LOG_DIR"
  mkdir -p $LOG_DIR
fi

echo $CMD
echo "LOG: $LOG_FILE"

nohup $CMD > $LOG_FILE 2>&1 &
sleep 1
tail -f $LOG_FILE
