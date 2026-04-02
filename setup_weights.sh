#!/usr/bin/env bash
# setup_weights.sh - Downloads pre-trained weights from Google Drive
set -e

FILE_ID="1EL4UmK5QDfc8CrBhunN8P-xHqFs4eJvv"
FILENAME="runs.tar.gz"

if [ -d "runs" ]; then
    echo "runs/ already exists. Skipping download."
    exit 0
fi

if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install --user gdown
fi

echo "Downloading pre-trained weights (~2.7 GB)..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$FILENAME" --fuzzy

echo "Extracting..."
tar -xzf "$FILENAME"

echo "Cleaning up..."
rm "$FILENAME"

echo "Done! runs/ is ready."
