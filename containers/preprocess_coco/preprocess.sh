#!/bin/bash

set -ex

if [ "$#" -ne 1 ]; then
    echo "Usage: ./preprocess.sh  bucket-name"
    exit
fi

DATA_BUCKET=$1

cd /tensorflow_tpu_models/tools/datasets
bash download_and_preprocess_coco.sh /scratch-dir
gsutil -m cp /scratch-dir/*.tfrecord ${DATA_BUCKET}/coco
gsutil cp /scratch-dir/raw-data/annotations/*.json ${DATA_BUCKET}/coco