#!/bin/bash

# Azure Container Registry details
ACR_NAME="prateekmlops23" 
ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"

docker login myregistry.azurecr.io -u <username> -p <password>

# Build Base Docker image
echo "Building Docker image..."
docker build -t base -f docker/base .

# Tag the image for ACR
ACR_IMAGE="${ACR_LOGIN_SERVER}/base:latest"
docker tag base $ACR_IMAGE

# Push the image to Azure Container Registry
echo "Pushing Docker image to Azure Container Registry..."
docker push $ACR_IMAGE

# Build Base Docker image
echo "Building Docker image..."
docker build -t digits -f docker/digits .

# Tag the image for ACR
ACR_IMAGE="${ACR_LOGIN_SERVER}/digits:latest"
docker tag digits $ACR_IMAGE

# Push the image to Azure Container Registry
echo "Pushing Docker image to Azure Container Registry..."
docker push $ACR_IMAGE

echo "Process complete."
