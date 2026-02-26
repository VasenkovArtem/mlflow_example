FROM python:3.12-slim

RUN apt-get update && apt-get install -y git curl wget

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app
