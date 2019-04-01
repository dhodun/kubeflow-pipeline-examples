#!/usr/bin/env bash


set -ex

if [ "$#" -ne 1 ]; then
    echo "Usage: ./train.sh  bucket-name"
    exit
fi

STORAGE_BUCKET=$1



echo '{"outputs": [{"source": "'$GCS_MODEL_DIR'", "type": "tensorboard"}]}' >> /mlpipeline-ui-metadata.json

cd /tpu/models/experimental/mask_rcnn


# removing error out because gsutil returns CommandException if directory is empty, wasn't able to quiet this
set +e
gsutil -q rm -rf gs://maskrcnn-kfp/mask-rcnn-model/*
set -e

python /tpu/models/experimental/mask_rcnn/mask_rcnn_main.py \
    --use_tpu=False \
    --model_dir="gs://maskrcnn-kfp/mask-rcnn-model" \
    --config="config.yaml" \
    --mode="train_and_eval" \
    --iterations_per_loop=1
