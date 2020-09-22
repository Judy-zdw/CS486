FROM ubuntu
WORKDIR /workspace
RUN apt update
RUN apt install -y python3.6 python3-pip
RUN pip3 install tensorflow numpy IPython
