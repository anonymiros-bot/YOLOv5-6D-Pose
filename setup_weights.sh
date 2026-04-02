#!/usr/bin/env bash
# setup_weights.sh - Downloads and extracts pre-trained weights
set -e

FILE_ID="1EL4UmK5QDfc8CrBhunN8P-xHqFs4eJvv"
FILENAME="runs.tar.gz"

if [ -d "runs" ]; then
    echo "runs/ already exists. Skipping."
    exit 0
fi

if [ ! -f "$FILENAME" ]; then
    echo "Attempting to download pre-trained weights (~2.7 GB)..."

    if ! command -v gdown &> /dev/null; then
        pip install --user gdown
    fi

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
fi

echo "Extracting..."
tar -xzf "$FILENAME"

echo "Cleaning up..."
rm "$FILENAME"

echo "Done! runs/ is ready."