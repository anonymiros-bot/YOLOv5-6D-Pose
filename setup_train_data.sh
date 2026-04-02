#!/usr/bin/env bash
# setup_train_data.sh - Downloads and extracts Unreal training images
set -e

FILE_ID="1H4CEpFuswE7IigmcYTkKquu8ckmK7BTK"
FILENAME="unreal_train_data.tar.gz"

if [ -d "data/unreal" ]; then
    echo "data/unreal/ already exists. Skipping."
    exit 0
fi

if [ ! -f "$FILENAME" ]; then
    echo "Attempting to download Unreal training images (~15 GB)..."

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

echo "Done! Training data is ready in data/unreal/."