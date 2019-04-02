FROM gcr.io/tensorflow/tpu-models:r1.12


RUN sudo apt-get update
# have to set this to get noninteractive install of python-tk without setting timezone, uses UTC?
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python-tk python-dev libgtk2.0-dev
RUN pip install --upgrade pip
RUN pip install Cython matplotlib \
    'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI' \
    opencv-python-headless \
    pyyaml

RUN git clone https://github.com/tensorflow/tpu/ /tpu
RUN cd /tpu && git pull && git reset --hard cfde494b3b0098afe272efb16124883ea3cbd38f