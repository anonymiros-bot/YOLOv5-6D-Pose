#!/usr/bin/env bash
# setup_data.sh - Downloads and extracts benchmark data into existing data/ directory
set -e

FILENAME="data.tar.zst"
FILE_ID="179BaZZiU6lcHpQI2ritt7IL802FehSvc"

if [ -d "data/sequences" ] && [ -d "data/data/mavic2" ] && [ -d "data/data/phantom4" ]; then
    echo "Benchmark data already exists. Skipping."
    exit 0
fi

if [ ! -f "$FILENAME" ]; then
    echo "Attempting to download benchmark data (~70 GB)..."

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

echo "Extracting into data/..."
mkdir -p data
tar --zstd -xf "$FILENAME" --strip-components=1 -C data

echo "Cleaning up..."
rm "$FILENAME"

echo "Done! Benchmark data is ready in data/."
