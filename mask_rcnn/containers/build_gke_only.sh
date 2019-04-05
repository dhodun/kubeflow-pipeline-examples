#!/bin/bash

#TODO: Build base image first?


cd containers/mask_rcnn_base
bash ./build.sh
cd ../train_mask_rcnn
bash ./build.sh