#!/usr/bin/env bash
./build.sh ; docker run -t -v ~/.config/gcloud:/root/.config/gcloud export-mask-rcnn-saved:latest jpeg gs://dhodun1-central1/mask-rcnn-model/job_190402_030903 \
 gs://dhodun1-central1/mask-rcnn-model/job_190402_030903;
docker run -t -v ~/.config/gcloud:/root/.config/gcloud export-mask-rcnn-saved:latest tensor gs://dhodun1-central1/mask-rcnn-model/job_190402_030903 \
 gs://dhodun1-central1/mask-rcnn-model/job_190402_030903