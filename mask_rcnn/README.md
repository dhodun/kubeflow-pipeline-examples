# How to train Mask-RCNN on TPUs with Kubeflow Pipelines (or just k8s)

## Setup

## Build the Docker Container

## Run the pipeline

# (Optional) Run Natively on GKE


## Setup

### Enable the appropriate APIs

Set your project

```bash
PROJECT=dhodun1
gcloud config set project $PROJECT
```

```bash
#TODO: test exactly which services are required in new project
SERVICES=(compute.googleapis.com storage-api.googleapis.com storage-component.googleapis.com tpu.googleapis.com \
 iam.googleapis.com tpu.googleapis.com)

gcloud services enable ${SERVICES[@]} --async
```


### Create a Kubernetes Cluster

Run the following:

```bash
CLUSTERNAME=mykfp3
```

Choose a zone that has the TPU version and pod size you want and set it:
https://cloud.google.com/tpu/docs/regions

```bash
ZONE=us-central1-a
```

(TODO: confirm TPU commands still needed at startup)
 
Create GKE cluster - note the 2 lines regarding TPU. These are required at Cluster creation

n1-standard-8 is based on TPU guide recommendations

```bash
gcloud config set compute/zone $ZONE
gcloud container clusters create $CLUSTERNAME \
  --cluster-version 1.11.7-gke.12 --enable-autoupgrade \
  --enable-autorepair \
  --zone $ZONE \
  --scopes cloud-platform \
  --enable-cloud-logging \
  --enable-cloud-monitoring \
  --machine-type n1-standard-8 \
  --enable-autoscaling --max-nodes=10 --min-nodes=2 \
  --enable-ip-alias \
  --enable-tpu 
```

## Build the Docker Containers

```bash
bash ./containers/build_gke_only.sh 
```

## Run the Container

First make sure you kubectl is pointing to the appropriate GKE cluster. You should not see any pods if this is a new cluster.

```bash
gcloud container clusters get-credentials $CLUSTERNAME
kubectl cluster-info
kubectl get pods
```

Create the TPU training pod which.

```bash
kubectl create -f gke_tpu.yaml
```

```bash
kubectl get pods
POD_NAME=$(kubectl get pods --sort-by=.metadata.creationTimestamp -o jsonpath="{.items[0].metadata.name}")
```

Take a look and see that the pod is pending because GKE is provisioning a TPU. At this point you should be able to see the TPU provisioninig in google Console.
https://console.cloud.google.com/compute/tpus

```bash
kubectl describe pod $POD_NAME
```

```bash
gcloud compute tpus list
```

We can monitor the creation of this TPU then the subsequent pod scheduling. This should take about 5 minutes.

```bash
watch kubectl describe pod $POD_NAME
```

Once the TPU has been created and the pod has been scheduled and created, we can stream the logs off to monitor training.
(TODO: switch this to -o json or get -o wide to support more c/129988998)
(TODO: show the TPU has annotated the pod with variable, etc once created)
```bash
kubectl get tpu -o wide
kubectl logs -f $POD_NAME
```