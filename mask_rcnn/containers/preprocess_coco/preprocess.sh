#!/bin/bash

set -ex

if [ "$#" -ne 1 ]; then
    echo "Usage: ./preprocess.sh  bucket-name"
    exit
fi

DATA_BUCKET=$1

cd /tpu/tools/datasets && \
bash download_and_preprocess_coco.sh ./data/dir/coco
gsutil -m cp ./data/dir/coco/*.tfrecord ${DATA_BUCKET}/coco && \
gsutil cp ./data/dir/coco/raw-data/annotations/*.json ${DATA_BUCKET}/coco
