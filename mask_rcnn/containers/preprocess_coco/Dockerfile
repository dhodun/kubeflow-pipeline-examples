

FROM gcr.io/dhodun1/mask-rcnn-base:latest

RUN mkdir /app
WORKDIR /app

COPY preprocess.sh /app

ENTRYPOINT ["bash", "/app/preprocess.sh"]
