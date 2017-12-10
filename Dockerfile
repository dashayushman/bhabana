FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

MAINTAINER Ayushman Dash <dash.ayushman.99@gmail.com>

# Install bare minimun requirements
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    curl \
    vim \
    wget \
    libssl-dev \
    openssl \
    doxygen

RUN apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.5.0/Python-3.5.0.tgz && \
    tar xzvf Python-3.5.0.tgz && \
    cd Python-3.5.0 && \
    ./configure && \
    make && \
    sudo make install

RUN echo 'alias python=python3' >> ~/.bash_aliases && \
    source ~/.bash_aliases

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py


RUN git clone https://github.com/dashayushman/bhabana.git ~/projects/bhabana && \
    cd ~/projects/bhabana/ && \
    pip install -r requirements/dev_requirements.txt && \
    cd ~/

RUN python -m spacy download en_vectors_web_lg
RUN python -m spacy download en_vectors_web_lg


EXPOSE 6006 8888

WORKDIR "/root"
CMD ["/bin/bash"]


