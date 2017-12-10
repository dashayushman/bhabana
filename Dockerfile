FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

MAINTAINER Ayushman Dash <dash.ayushman.99@gmail.com>

# Install bare minimun requirements
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		pkg-config \
		software-properties-common \
		unzip \
		vim \
		wget \
		python3 \
		python3-dev \
		python3-tk \
		python3-numpy \
		doxygen \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py


RUN git clone https://github.com/dashayushman/bhabana.git /root/bhabana && \
    cd /root/bhabana && \
    pip3 install -r requirements/dev_requirements.txt && \
    cd /

EXPOSE 6006 8888

WORKDIR "/root"
CMD ["/bin/bash"]


