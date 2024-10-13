#!/bin/bash

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

# Webhook URL for notifications (replace with your actual webhook URL)
WEBHOOK_URL=https://hooks.example.com/services/your-webhook-path

# Capture start time
start_time=$(date +%s)

# Define an associative array to hold container names and their respective port mappings
declare -A containers
containers["user1_env"]="7010 7110"
containers["user2_env"]="7007 7107"
# ... Add more container definitions as needed

echo "Creating network share directories on the host..."
mkdir -p /home/network-share1 /home/network-share2 /home/network-share3 /home/network-share4
chmod 755 /home/network-share1 /home/network-share2 /home/network-share3 /home/network-share4

echo "Stopping and removing specified containers..."
for container in "${!containers[@]}"
do
  docker stop "$container" && docker rm "$container"
done

# Build the new Docker image
echo "Building new Docker image..."
if docker build -t cuda-enabled-env . ; then
  echo "Docker image built successfully."
else
  echo "Docker image build failed. Aborting."
  message="Docker build failed. Please check the build logs."
  python3 send_notification.py "#alerts" "$message" "$WEBHOOK_URL" 
  exit 1
fi

echo "Recreating containers with GPU support, new volume mounts, and necessary capabilities..."
for container in "${!containers[@]}"
do
  read -r ssh_port pass_through_port <<< "${containers[$container]}"
  docker run -d --privileged --gpus all \
    --shm-size=8gb \
    --cap-add=SYS_ADMIN \
    --cap-add=MKNOD \
    --cap-add=SYS_RESOURCE \
    --cap-add DAC_READ_SEARCH \
    --device=/dev/fuse \
    -p "$ssh_port:22" \
    -p "$pass_through_port:$pass_through_port" \
    -v "/home/dockerdata/$container:/dockerdata" \
    -v "/home/shared-data:/shared-data:rw" \
    --restart unless-stopped \
    --name "$container" cuda-enabled-env

  # Mount the network shares inside the container
  docker exec "$container" mkdir -p /network-share1 /network-share2 /network-share3 /network-share4
  docker exec "$container" mount -t cifs -o credentials=/root/.network-creds,uid=0,gid=0,ro,dir_mode=0777,file_mode=0777,rsize=131072,wsize=131072,actimeo=0,vers=3.0 //server-ip/network-share1 /network-share1
  docker exec "$container" mount -t cifs -o credentials=/root/.network-creds,uid=0,gid=0,ro,dir_mode=0777,file_mode=0777,rsize=131072,wsize=131072,actimeo=0,vers=3.0 //server-ip/network-share2 /network-share2
  docker exec "$container" mount -t cifs -o credentials=/root/.network-creds,uid=0,gid=0,ro,dir_mode=0777,file_mode=0777,rsize=131072,wsize=131072,actimeo=0,vers=3.0 //server-ip/network-share3 /network-share3
  docker exec "$container" mount -t cifs -o credentials=/root/.share1-creds,uid=0,gid=0,rw,dir_mode=0777,file_mode=0777,rsize=131072,wsize=131072,actimeo=0,vers=3.0 //server-name/network-share4 /network-share4
done

echo "Containers have been recreated with GPU support, new volume mounts, and necessary capabilities."

# Test GPU accessibility
echo "Testing GPU accessibility with a temporary CUDA container..."
docker run --rm --privileged --gpus all nvidia/cuda:12.2.2-devel-ubuntu22.04 nvidia-smi

# Network share mount test
echo "Testing network share mount in a temporary container..."
docker run --rm --privileged cuda-enabled-env bash -c "\
  mkdir -p /test-mount && \
  mount -t cifs -o credentials=/root/.network-creds,uid=0,gid=0,ro,dir_mode=0777,file_mode=0777,rsize=131072,wsize=131072,actimeo=0,vers=3.0 //server-ip/network-share3 /test-mount && \
  file_count=\$(ls /test-mount | wc -l) && \
  echo \"Number of files found: \$file_count\" && \
  if [ \$file_count -lt 2 ]; then \
    echo \"File count is less than 2. Network share mount test failed.\" && \
    umount /test-mount && \
    exit 1; \
  else \
    echo \"File count is 2 or more. Network share mount test passed.\" && \
    umount /test-mount && \
    exit 0; \
  fi \
" && MOUNT_TEST_RESULT="passed" || MOUNT_TEST_RESULT="failed"

# Calculate the duration
end_time=$(date +%s)  
duration=$((end_time - start_time))

if [ "$MOUNT_TEST_RESULT" = "passed" ]; then
  echo "Network share mount test completed successfully."
  message="Docker build, container recreation, GPU test, and network share mount test completed successfully in $duration seconds."
else
  echo "Network share mount test failed."
  message="Docker build and container recreation completed but network share mount test failed in $duration seconds."
fi

# Send the notification
# Assuming send_notification.py is in the same directory as this script
python3 send_notification.py "#log-channel" "$message" "$WEBHOOK_URL"

echo "Build and setup process completed."
