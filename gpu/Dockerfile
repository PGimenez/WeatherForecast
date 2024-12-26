FROM ubuntu:22.04

# Install basic tools and Intel GPU tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    intel-gpu-tools \
    pciutils \
    lshw \
    ffmpeg \
    vainfo \
    intel-media-va-driver-non-free \
    i965-va-driver-shaders \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for hardware acceleration
ENV LIBVA_DRIVER_NAME=i965 \
    LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri

# Create a non-root user
RUN useradd -u 1000 -m gpuuser && \
    usermod -a -G video,render gpuuser

USER gpuuser
WORKDIR /home/gpuuser

CMD ["sleep", "infinity"]
