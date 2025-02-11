FROM bitnami/spark:latest

USER root

# install netcat
RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

USER spark