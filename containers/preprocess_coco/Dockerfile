
FROM gcr.io/tensorflow/tpu-models:r1.12

RUN mkdir /app
WORKDIR /app

COPY preprocess.sh /app

ENTRYPOINT ["bash", "/app/preprocess.sh"]
