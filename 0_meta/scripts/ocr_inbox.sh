#!/bin/bash

# OCR Inbox Script
# Scans an inbox directory for images and generates .txt sidecars using Tesseract.

INBOX_DIR="${1:-./1_inbox}"

# Ensure tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "Error: tesseract not found. Please install it with 'brew install tesseract'."
    exit 1
fi

echo "Scanning $INBOX_DIR for images..."

# Find images case-insensitively
find "$INBOX_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tiff" \) | while read -r img; do
    base="${img%.*}"
    txt="$base.txt"
    
    # Skip if .txt exists and is newer than image
    if [ -f "$txt" ] && [ "$txt" -nt "$img" ]; then
        continue
    fi
    
    echo "OCR-ing: $(basename "$img")"
    tesseract "$img" "$base" -l eng quiet
done

echo "OCR check complete."
