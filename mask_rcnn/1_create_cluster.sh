#!/bin/bash

CLUSTERNAME=mykfp
ZONE=us-central1-b

gcloud config set compute/zone $ZONE
gcloud beta container clusters create $CLUSTERNAME \
  --cluster-version 1.11.2-gke.18 --enable-autoupgrade \
  --zone $ZONE \
  --scopes cloud-platform \
  --enable-cloud-logging \
  --enable-cloud-monitoring \
  --machine-type n1-standard-2 \
  --num-nodes 4 \
  --enable-tpu \
  --enable-ip-alias

#  --preemptible \
#  --enable-autoprovisioning --max-cpu=40 --max-memory=1024 \
#  --enable-autoscaling --max-nodes=10 --min-nodes=3

kubectl create clusterrolebinding ml-pipeline-admin-binding --clusterrole=cluster-admin --user=$(gcloud config get-value account)

kubectl create clusterrolebinding pipelinerunnerbinding \
  --clusterrole=cluster-admin \
  --serviceaccount=kubeflow:pipeline-runner

#TODO: create auto-node-pool and large limit on cores

	#gcloud beta container clusters update dhodunpipeline --enable-autoprovisioning \
	 #       --max-cpu 300 --max-memory 1000