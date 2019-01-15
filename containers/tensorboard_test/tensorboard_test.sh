#!/usr/bin/env bash

tb_dir='gs://dhodun1-central1/mask-rcnn-model/job_20181225_225751'

echo '{"outputs": [{"source": "'$tb_dir'", "type": "tensorboard"}]}' >> /mlpipeline-ui-metadata.json