#!/bin/bash

# OCR Single File Script
# Performs OCR on a single file and creates a sidecar .txt.

FILE="$1"

if [ -z "$FILE" ]; then
    echo "Usage: $0 <path-to-image>"
    exit 1
fi

if [ ! -f "$FILE" ]; then
    echo "Error: File $FILE not found."
    exit 1
fi

# Ensure tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "Error: tesseract not found. Please install it with 'brew install tesseract'."
    exit 1
fi

base="${FILE%.*}"
echo "OCR-ing: $(basename "$FILE")..."
tesseract "$FILE" "$base" -l eng quiet
echo "Created: $(basename "$base.txt")"
