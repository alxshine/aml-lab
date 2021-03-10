FROM tensorflow/tensorflow

RUN apt-get update -q -y && apt-get install -q -y git

# by creating a dev user I can directly use this image in devcontainers
RUN adduser --disabled-password --gecos "" dev
USER dev