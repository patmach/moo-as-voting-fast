FROM python:3.9.10-slim-buster

RUN python -m pip install --upgrade pip

RUN pip install matplotlib==3.5.1
RUN pip install caserecommender==1.1.1
RUN pip install mlflow==1.23.1
RUN pip install flask==2.0.3

RUN apt-get update
RUN apt-get install -y git
