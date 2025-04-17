#!/bin/bash
set -e

# 1. Install required dependency packages
echo "Installing required dependencies..."
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# 2. Install Docker GPG key
echo "Installing Docker GPG key..."
curl -fsSL https://download.docker.com/linux/raspbian/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null

# 3. Add the Docker repository
echo "Adding Docker repository..."
echo "deb [arch=armhf signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/raspbian $(lsb_release -c | awk '{print $2}') stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. Update the package list
echo "Updating package list..."
sudo apt update

# 5. Install Docker CE, Docker CLI, and Containerd (docker.io package) along with docker-compose
echo "Installing Docker..."
sudo apt install docker.io && sudo apt install docker-compose

# 6. Allow running Docker as a non-root user by adding the current user to the docker group
echo "Adding user to the docker group..."
sudo usermod -aG docker $USER

# 7. Start and enable the Docker service
echo "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# 8. Verify Docker installation
echo "Verifying Docker installation..."
docker --version

# 9. Provide instructions for next steps
echo "Docker installation complete!"
echo "You need to log out and log back in, or run 'newgrp docker' to apply the group changes."
echo "After that, you can run Docker without 'sudo'."
