FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

RUN git config --global user.email "272702@student.pwr.edu.pl" && \
    git config --global user.name "Mikołaj Olesiński"