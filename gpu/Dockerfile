# Start with Ubuntu base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Configure dpkg to automatically accept maintainer's config file versions
RUN echo 'Dpkg::Options {"--force-confnew"};' > /etc/apt/apt.conf.d/local

# Update package lists and install essential packages first
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    netbase \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Install Python and development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-dev \
    python3-pip \
    vim \
    git \
    curl \
    wget \
    build-essential \
    pkg-config \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Intel GPU dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    intel-media-va-driver-non-free \
    libmfx1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Intel GPU
ENV ONEAPI_DEVICE_SELECTOR=gpu
ENV NEOReadDebugKeys=1
ENV ClDeviceGlobalMemSizeAvailable=1

# Upgrade pip and install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    intel-tensorflow \
    tensorflow \
    intel-extension-for-tensorflow \
    numpy \
    pandas \
    scikit-learn \
    jupyterlab \
    matplotlib \
    ipython

# Set Python3 as default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Default command
CMD ["bash"]
