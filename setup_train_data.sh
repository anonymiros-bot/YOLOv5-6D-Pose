#!/usr/bin/env bash
# setup_train_data.sh - Downloads Unreal training images
set -e

FILE_ID="1WpraNktn7V5RTxr9kezKYLwCgV50uEUh"
FILENAME="train_data.tar.gz"

# What does this extract to? Need to know the target directory
# to check if it already exists
if [ -d "data/SOMETHING" ]; then
    echo "Training data already exists. Skipping."
    exit 0
fi

if ! command -v gdown &> /dev/null; then
    pip install --user gdown
fi

echo "Attempting to download Unreal training images (~X.X GB)..."

if gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$FILENAME" --fuzzy; then
    echo "Download successful."
else
    echo ""
    echo "Automatic download failed."
    echo "Please download manually from:"
    echo "  https://drive.google.com/file/d/${FILE_ID}"
    echo "Place ${FILENAME} in the repository root and re-run this script."
    exit 1
fi

echo "Extracting..."
tar -xzf "$FILENAME"

echo "Cleaning up..."
rm "$FILENAME"

echo "Done! Training data is ready."
