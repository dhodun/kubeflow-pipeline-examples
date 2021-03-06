#!/bin/bash

echo 'connect to http://localhost:8085/pipeline'

export NAMESPACE=kubeflow
kubectl port-forward -n ${NAMESPACE} $(kubectl get pods -n ${NAMESPACE} --selector=service=ambassador -o jsonpath='{.items[0].metadata.name}') 8085:80

# http://localhost:8085/pipeline