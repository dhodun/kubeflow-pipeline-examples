#!/usr/bin/env bash

#TODO: consolidate TPU and CPU directories

set -ex

echo $1
echo $2

if [ "$#" -ne 2 ]; then
    echo "Usage: ./train.sh  bucket-name  train-data-dir"
    exit
fi

#TODO: error checking for gs:// vs non
#TODO: implement non-coco conversion (so you can bring your own data)

STORAGE_BUCKET=$1
TRAIN_DATA_DIR=$2
TRAIN_JOB_NAME=job_$(date -u +%y%m%d_%H%M%S)
MODEL_DIR=$1/mask-rcnn-model/$TRAIN_JOB_NAME

echo '{"outputs": [{"source": "'$MODEL_DIR'", "type": "tensorboard"}]}' >> /mlpipeline-ui-metadata.json

sed -i "s/{BUCKET_NAME}/$STORAGE_BUCKET/g" /tpu/models/experimental/mask_rcnn/config.yaml

cd /tpu/models/experimental/mask_rcnn

python /tpu/models/experimental/mask_rcnn/mask_rcnn_main.py \
    --use_tpu=True \
    --model_dir=$MODEL_DIR \
    --config="config-cpu-test.yaml" \
    --mode="train_and_eval" \
    --iterations_per_loop=1

echo $MODEL_DIR > /output.txt
