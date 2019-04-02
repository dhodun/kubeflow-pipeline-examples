#!/usr/bin/env bash

#TODO: consolidate TPU and CPU directories

set -ex

if [ "$#" -ne 1 ]; then
    echo "Usage: ./train.sh  bucket-name"
    exit
fi

STORAGE_BUCKET=$1
TRAIN_JOB_NAME=job_$(date -u +%y%m%d_%H%M%S)
MODEL_DIR=$1/mask-rcnn-model/$TRAIN_JOB_NAME

echo '{"outputs": [{"source": "'$MODEL_DIR'", "type": "tensorboard"}]}' >> /mlpipeline-ui-metadata.json

sed -i "s/{BUCKET_NAME}/$STORAGE_BUCKET/g" /tpu/models/experimental/mask_rcnn/config.yaml

cd /tpu/models/experimental/mask_rcnn

python /tpu/models/experimental/mask_rcnn/mask_rcnn_main.py \
    --use_tpu=True \
    --model_dir=$MODEL_DIR \
    --config="config.yaml" \
    --mode="train_and_eval" \
    --iterations_per_loop=1

echo $MODEL_DIR > /output.txt
