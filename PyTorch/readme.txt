Simple Python ML Model Dockerfile:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY train.py .

RUN mkdir -p /app/data

CMD ["python", "train.py"]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

docker build -t mnist-pytorch .
docker run --rm mnist-pytorch