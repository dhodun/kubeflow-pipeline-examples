#!/bin/bash

set -e

python3 mask_rcnn.py

cd containers/train_mask_rcnn_cpu
bash ./build.sh
cd ../export_mask_rcnn_saved
bash ./build.sh
cd ../..

python3 run_pipeline.py