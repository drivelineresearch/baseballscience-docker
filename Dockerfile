# MIT License
#
# Copyright (c) 2024 Driveline Research
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Kyle Boddy, Driveline Research

# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set non-interactive installation mode for apt-get and initial PATH
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:${PATH}

# Set the workspace folder and MAX_JOBS for ninja/flash-attn
WORKDIR /workspace
ENV MAX_JOBS=6

# Install system packages
RUN apt-get update && apt-get install -y \
    openssh-server sudo vim git curl wget ffmpeg libsm6 libxext6 libxrender-dev \
    python3-pip python3-dev build-essential libopencv-dev libssl-dev zlib1g-dev \
    cudnn9-cuda-12 libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
    libswresample-dev libswscale-dev python3-venv btop nvtop nano net-tools htop \
    zip unzip git-lfs tmux screen glances cmake software-properties-common \
    && add-apt-repository ppa:ondrej/php \
    && apt-get update \
    && apt-get install -y php8.0 php8.0-cli php8.0-common php8.0-mysql php8.0-xml \
    php8.0-mbstring php8.0-curl php8.0-gd libmariadb-dev libmariadb-dev-compat \
    ninja-build jq python3-tk libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CIFS utilities for mounting network shares
RUN apt-get update && apt-get install -y cifs-utils && rm -rf /var/lib/apt/lists/*

# Unminimize the system
RUN yes | unminimize

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

# Set PATH to include Miniconda
ENV PATH=/opt/miniconda/bin:${PATH}

# Add PATH setting for Conda and CUDA to .bashrc and .profile
RUN echo "export PATH=/opt/miniconda/bin:/usr/local/cuda/bin:${PATH}" >> /root/.bashrc && \
    echo "export PATH=/opt/miniconda/bin:/usr/local/cuda/bin:${PATH}" >> /root/.profile

# Set up Conda and append conda-forge to the channels
RUN conda config --append channels conda-forge

# Install ezc3d using conda
RUN conda install -c conda-forge ezc3d

# Install NodeJS
RUN apt-get update && apt-get install -y ca-certificates curl gnupg \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg \
    && NODE_MAJOR=20 \
    && echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install nodejs -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up SSH for remote connections
RUN mkdir /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
EXPOSE 22

# Install Python packages and database connectors using Conda
RUN /opt/miniconda/bin/pip install opencv-python ipykernel xgboost lightgbm pandas flask scikit-learn scipy numpy seaborn matplotlib build nvitop ffmpeg-python Jinja2 imageio[ffmpeg] mysql-connector-python PyMySQL torch transformers datasets deepspeed bitsandbytes exllamav2 tokenizers sentencepiece optuna ultralytics supervision mariadb

# Install additional Python packages
RUN /opt/miniconda/bin/pip install dlib tensorflow
RUN /opt/miniconda/bin/pip install face-recognition flash-attn

# Install R and R packages using Conda
RUN conda install -c r r-base \
    && conda install -c r r-ggplot2 r-dplyr r-tidyr r-shiny r-caret r-randomForest r-rmarkdown r-keras r-tensorflow r-imager r-xgboost r-lightgbm r-h2o r-RMySQL r-RMariaDB

# Copy credentials (ensure these files are properly sanitized before sharing)
COPY .network-creds /root/.network-creds
COPY .share1-creds /root/.share1-creds
COPY .share2-creds /root/.share2-creds

# Set permissions for credentials
RUN chmod 600 /root/.network-creds /root/.share1-creds /root/.share2-creds

# Create mount points
RUN mkdir -p /network-share1 /network-share2 /network-share3 /network-share4

# Adjust system limits
RUN echo "fs.inotify.max_user_watches = 1048576" >> /etc/sysctl.conf \
    && echo "fs.file-max = 2097152" >> /etc/sysctl.conf \
    && echo "vm.min_free_kbytes = 65536" >> /etc/sysctl.conf \
    && echo "vm.vfs_cache_pressure = 50" >> /etc/sysctl.conf \
    && sysctl -p
    
# Copy entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Apply system limits
RUN echo '* hard nofile 1048576' >> /etc/security/limits.conf && \
    echo '* soft nofile 1048576' >> /etc/security/limits.conf && \
    echo 'fs.file-max = 2097152' >> /etc/sysctl.conf && \
    echo 'vm.min_free_kbytes = 65536' >> /etc/sysctl.conf && \
    echo 'fs.inotify.max_user_watches = 1048576' >> /etc/sysctl.conf && \
    sysctl -p

# Reinforce PATH
ENV PATH=/opt/miniconda/bin:/usr/local/cuda/bin:${PATH}

# Reset the frontend variable and start the SSH service
ENV DEBIAN_FRONTEND=
CMD ["/usr/sbin/sshd", "-D"]
