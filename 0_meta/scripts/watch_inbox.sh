#!/bin/bash

# Watch Inbox Script (Fallback)
# Polls the inbox every 30 seconds to run OCR.

INBOX_DIR="${1:-./1_inbox}"
SCRIPT_DIR="$(dirname "$0")"
OCR_SCRIPT="$SCRIPT_DIR/ocr_inbox.sh"

echo "Watching $INBOX_DIR for new images (Polling every 30s)..."
echo "Note: For better performance on macOS, use an Automator Folder Action or 'Hazel'."

while true; do
    bash "$OCR_SCRIPT" "$INBOX_DIR"
    sleep 30
done
