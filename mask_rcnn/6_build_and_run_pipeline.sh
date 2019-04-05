#!/bin/bash

set -e


if [ "$#" -ne 3 ]; then
    echo "Usage: 6_build_and_run_pipeline.sh train_devicetype isTest"
    exit
fi

TRAIN_DEVICE=$1
IS_TEST=$2

python3 mask_rcnn.py


cd containers/preprocess_coco
bash ./build.sh
cd ../train_mask_rcnn
bash ./build.sh
cd ../train_mask_rcnn_tpu
bash ./build.sh
cd ../export_mask_rcnn_saved
bash ./build.sh
cd ../..

python3 run_pipeline.py $1 $2