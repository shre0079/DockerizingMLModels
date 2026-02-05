Simple Python ML Model Dockerfile:

~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY TestML-Model.py .

RUN mkdir -p output

CMD ["python", "TestML-Model.py"]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

docker build -t simple-ml-app .
docker run --rm -v ${PWD}/output:/app/output simple-ml-app

~~~~~~~~~~~~~~~~~~~~~~~~~~~~
