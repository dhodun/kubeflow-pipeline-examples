
FROM gcr.io/dhodun1/mask-rcnn-base:latest


# this command has been moved into the container to accomodate TPU training in same container
#RUN pip install tf-nightly==1.14.1.dev20190319


RUN mkdir /app
WORKDIR /app


COPY train.sh /app
COPY test.sh /app
COPY config-cpu-test.yaml /tpu/models/experimental/mask_rcnn/
COPY config-tpu-test.yaml /tpu/models/experimental/mask_rcnn/
COPY extract_metrics.py /tpu/models/experimental/mask_rcnn/
COPY test.py /tpu/models/experimental/mask_rcnn

ENTRYPOINT ["bash", "/app/train.sh"]
