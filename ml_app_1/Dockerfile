FROM pytorch/torchserve:0.12.0-cpu

ARG USERNAME=model-server

COPY config.properties /home/model-server/
COPY model_store/ /home/model-server/model-store/

USER root

RUN apt-get update && apt-get -y install curl

RUN pip install ultralytics

RUN ls -lah /home/model-server/model-store/

USER ${USERNAME}