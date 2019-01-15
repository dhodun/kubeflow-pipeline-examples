#!/usr/bin/env bash
./build.sh ; docker run -t --entrypoint=/bin/bash train-mask-rcnn:latest  /app/test.sh dhodun1-central1