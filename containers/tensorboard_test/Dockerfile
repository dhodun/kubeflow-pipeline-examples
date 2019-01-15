
FROM gcr.io/tensorflow/tpu-models:r1.11

RUN mkdir /app

COPY tensorboard_test.sh /app

ENTRYPOINT ["bash", "/app/tensorboard_test.sh"]
