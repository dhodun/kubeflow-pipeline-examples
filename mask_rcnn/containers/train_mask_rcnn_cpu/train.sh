#!/usr/bin/env bash


set -ex

if [ "$#" -ne 3 ]; then
    echo "Usage: ./train.sh  bucket-name train-data-dir useTPU"
    exit
fi

STORAGE_BUCKET=$1
TRAIN_DATA_DIR=$2
USE_TPU=$3

TRAIN_JOB_NAME=job_$(date -u +%y%m%d_%H%M%S)
MODEL_DIR=$1/mask-rcnn-model/$TRAIN_JOB_NAME

echo '{"outputs": [{"source": "'$MODEL_DIR'", "type": "tensorboard"}]}' >> /mlpipeline-ui-metadata.json


# install nightly TF if running training on CPU

CONFIG_FILE='config-tpu.yaml'

if [[ ! $USE_TPU ]]
then
  echo "Installing tf-nightly for CPU training"
  pip install tf-nightly==1.14.1.dev20190319
  CONFIG_FILE='config-tpu.yaml'
fi



# uses pipe since variable has forward slashes, also logic to remove gs:// if it exists
sed -i "s|{BUCKET_NAME}|${STORAGE_BUCKET//gs:\/\/}|g" /tpu/models/experimental/mask_rcnn/CONFIG_FILE

cd /tpu/models/experimental/mask_rcnn


python /tpu/models/experimental/mask_rcnn/mask_rcnn_main.py \
    --use_tpu=$USE_TPU \
    --model_dir=$MODEL_DIR \
    --config=$CONFIG_FILE \
    --mode="train" \
    --iterations_per_loop=1

echo "Installing tf-nightly for CPU eval"
pip install tf-nightly==1.14.1.dev20190319

python /tpu/models/experimental/mask_rcnn/mask_rcnn_main.py \
    --use_tpu=$USE_TPU \
    --model_dir=$MODEL_DIR \
    --config=$CONFIG_FILE \
    --mode="eval" \
    --iterations_per_loop=1

# find the latest file in the job director
#TODO: find more reliable way to get this file
EVAL_FILE="$(gsutil ls -l $MODEL_DIR/eval/events* | sort -k 2 | head -n1 | awk '{ print $NF }')"

# extracts the metrics, dumps into eval_metrics.csv, and exports the 2 most important back into the pipeline
python /tpu/models/experimental/mask_rcnn/extract_metrics.py $EVAL_FILE

echo $MODEL_DIR > /model_dir.txt