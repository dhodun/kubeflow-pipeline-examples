#!/usr/bin/env bash


set -ex

if [ "$#" -ne 3 ]; then
    echo "Usage: ./save.sh  tensor|jpeg CHECKPOINT_PATH EXPORT_DIR"
    exit
fi

FORMAT=$1
#TODO: update to model_dir or at least add logic to find the latest checkpoint
CHECKPOINT_PATH=$2/model.ckpt-1
EXPORT_DIR=$3/$FORMAT

export PYTHONPATH=$PYTHONPATH:/tpu/models/

if [ "$FORMAT" == "jpeg" ]; then
    python /tpu/models/experimental/mask_rcnn/export_saved_model.py \
      --export_dir=$EXPORT_DIR \
      --checkpoint_path=$CHECKPOINT_PATH \
      --input_type="image_bytes" \
      --batch_size=1
elif [ "$FORMAT" == "tensor" ]; then
    python /tpu/models/experimental/mask_rcnn/export_saved_model.py \
      --export_dir=$EXPORT_DIR \
      --checkpoint_path=$CHECKPOINT_PATH \
      --input_type="image_tensor" \
      --batch_size=1 \
      --config="image_size=[876,1472]"
fi



echo $EXPORT_DIR > /output.txt