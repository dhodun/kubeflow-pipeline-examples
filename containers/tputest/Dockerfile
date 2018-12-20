
FROM gcr.io/tensorflow/tpu-models:r1.11

RUN mkdir /app

COPY cloud-tpu.py /app

ENTRYPOINT ["python", "/app/cloud-tpu.py"]
