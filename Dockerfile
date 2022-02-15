FROM tensorflow/tensorflow:latest-gpu

# Various Python and C/build deps
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget

RUN apt-get install -y pkg-config
RUN apt-get install -y libjpeg-dev
RUN apt-get install -y libpng-dev
RUN apt-get install -y libtiff-dev
RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y build-essential
RUN apt-get install -y cmake
RUN apt-get install -y git

RUN apt-get update
RUN apt-get install -y python3-opencv

RUN apt-get install -y python3-pip
RUN pip3 install opencv-python

RUN pip3 install matplotlib
RUN pip3 install kymatio
RUN pip3 install sklearn

CMD ["python3", "/src/scattering.py"]
# docker run -it --env="DISPLAY=$DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --gpus all --device=/dev/video0:/dev/video0 video
