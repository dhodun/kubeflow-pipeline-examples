#!/usr/bin/env bash


set -ex

if [ "$#" -ne 1 ]; then
    echo "Usage: ./train.sh  bucket-name"
    exit
fi

STORAGE_BUCKET=$1


export GCS_MODEL_DIR="gs://dhodun1-central1/mask-rcnn-model/job_$(date +%Y%m%d_%H%M%S)"
export RESNET_CHECKPOINT=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602
export RESNET_DEPTH=50
export PATH_GCS_MASKRCNN=${STORAGE_BUCKET}/coco
# export TPU_NAME=maskrcnn1

echo '{"outputs": [{"source": "'$GCS_MODEL_DIR'", "type": "tensorboard"}]}' >> /mlpipeline-ui-metadata.json

cd /tpu/models/experimental/mask_rcnn

python mask_rcnn_main.py \
    --use_tpu=True \
    --model_dir=${GCS_MODEL_DIR:?} \
    --resnet_checkpoint=${RESNET_CHECKPOINT} \
    --hparams="resnet_depth=${RESNET_DEPTH},use_bfloat16=true,learning_rate=0.00001,lr_warmup_init=0.00001" \
    --num_cores=8 \
    --train_batch_size=64 \
    --training_file_pattern="${PATH_GCS_MASKRCNN:?}/train-*" \
    --validation_file_pattern="${PATH_GCS_MASKRCNN:?}/val-*" \
    --val_json_file="${PATH_GCS_MASKRCNN:?}/instances_val2017.json" \
    --mode="train_and_eval" \
    --eval_batch_size=8 \
    --num_epochs=1 \
    --iterations_per_loop=10 \
    --num_examples_per_epoch=640 \
    --eval_samples=8
    #--num_examples_per_epoch=6400 \

