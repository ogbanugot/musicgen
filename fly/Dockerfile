FROM ubuntu:22.04
RUN apt update -q && apt install -y python3 python3-pip python3-venv python3-wheel git nano && \
    apt clean && rm -f /var/lib/apt/lists/*_*

ARG NONROOT_USER
RUN echo "User will be $NONROOT_USER"
ENV PYTHON_USER=$NONROOT_USER

# Create unprivileged user with a home dir and using bash
RUN useradd -ms /bin/bash $PYTHON_USER

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install build-essential libeigen3-dev libyaml-dev libfftw3-dev libtag1-dev libchromaprint-dev -y

RUN apt install screen -y

RUN apt install wget -y

RUN apt install curl -y

RUN apt-get update && apt-get -y install tmux

COPY --chmod=0755 ./entrypoint.sh ./entrypoint.sh
COPY --chown=$PYTHON_USER:$PYTHON_USER --chmod=0755 ./post-initialization.sh ./post-initialization.sh
# If you have a requirements.txt for the project, uncomment this and
# adjust post-initialization.sh to use it
# COPY --chown=$PYTHON_USER:$PYTHON_USER requirements.txt .

# CMD ["sleep", "inf"]
CMD ["/bin/bash", "-c", "./entrypoint.sh $PYTHON_USER"]
