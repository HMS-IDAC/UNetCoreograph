FROM python:3.12

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    --no-cache-dir \
    cython scikit-image matplotlib tifffile scipy opencv-python zarr tensorflow tf_keras

COPY . /app
