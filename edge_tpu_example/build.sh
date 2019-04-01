#!/usr/bin/env bash

set -e

DETECT_DIR=./edgetpu/detection && mkdir -p $DETECT_DIR

cd $DETECT_DIR

wget -O Dockerfile "http://storage.googleapis.com/cloud-iot-edge-pretrained-models/docker/obj_det_docker"

CONTAINER_NAME=detect-tutorial
TAG_NAME='latest'

docker build -t ${CONTAINER_NAME} .

PROJECT_ID=$(gcloud config config-helper --format "value(configuration.properties.core.project)")

docker tag ${CONTAINER_NAME} gcr.io/${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}
docker push gcr.io/${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}

