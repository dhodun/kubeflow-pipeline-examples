#!/usr/bin/env bash
./build.sh; docker run -t preprocess-coco:latest gs://maskrcnn-kfp
