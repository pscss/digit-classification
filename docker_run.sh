#!/bin/bash

# Step 1: Build Docker Image
echo "Building Docker image"
docker build -t dependency:v1 -f docker/dependency .

# Step 2: Run Docker Image
echo "Creating an empty 'save_models' directory"
mkdir save_models

echo "Listing contents of 'save_models' directory (before running Docker image)"
ls -lh save_models

echo "Running Docker image"
docker run -v "$(pwd)/save_models:/digits/models" digits:v1

echo "Listing contents of 'save_models' directory (after running Docker image)"
ls -lh save_models

# Step 3: Delete 'save_models' directory
echo "Deleting the 'save_models' directory"
rm -rf save_models