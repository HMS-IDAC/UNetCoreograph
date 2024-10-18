FROM python:3.12

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    --no-cache-dir \
    cython \
    imagecodecs \
    matplotlib \
    opencv-python \
    scikit-image \
    scipy \
    tensorflow \
    tf_keras \
    tifffile \
    zarr

COPY . /app
