
FROM gcr.io/dhodun1/mask-rcnn-base:latest



RUN pip install tf-nightly==1.14.1.dev20190319



RUN mkdir /app
WORKDIR /app


COPY save.sh /app

ENTRYPOINT ["bash", "/app/save.sh"]
