FROM tensorflow/tensorflow:1.15.0-py3

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libtiff5-dev

RUN pip install scikit-image==0.14.2 matplotlib tifffile==2020.2.16 pytiff==0.8.1 scipy==1.1.0 opencv-python

COPY . /app
