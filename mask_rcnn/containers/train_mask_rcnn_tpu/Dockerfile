
FROM gcr.io/dhodun1/mask-rcnn-base:latest


RUN mkdir /app
WORKDIR /app


COPY train.sh /app
COPY test.sh /app
COPY config.yaml /tpu/models/experimental/mask_rcnn/
COPY test.py /tpu/models/experimental/mask_rcnn

ENTRYPOINT ["bash", "/app/train.sh"]
