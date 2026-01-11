# OCR Setup & Workflow

This vault includes an automated OCR system to turn screenshots and images in `1_inbox` into searchable text sidecars.

## Prerequisites

1. **Homebrew**: Required to install the OCR engine.
2. **Tesseract**: The OCR engine itself. 
   - Install via terminal: `brew install tesseract`

## Included Scripts (`0_meta/scripts/`)

- **`ocr_inbox.sh`**: Scans the entire `1_inbox` folder and generates `.txt` files for any images missing them.
- **`ocr_once.sh <file>`**: Runs OCR on a single specific image file.
- **`watch_inbox.sh`**: A polling script that checks for new images every 30 seconds (fallback method).

## Automated Trigger (Recommended: macOS Automator)

For a seamless experience where OCR happens instantly when you drop a file:

1. Open **Automator.app**.
2. Create a new **Folder Action**.
3. At the top, select "Folder Action receives files and folders added to: **1_inbox**".
4. Add the action: **Run Shell Script**.
5. Set Shell to `/bin/bash` and Pass Input to **as arguments**.
6. Paste the following script:
   ```bash
   for f in "$@"
   do
       /Users/ljbaumer/Documents/AI_Working/0_meta/scripts/ocr_once.sh "$f"
   done
   ```
7. Save it as "Vault-Inbox-OCR".

## Manual Usage

To manually refresh the inbox:
```bash
bash 0_meta/scripts/ocr_inbox.sh
```

## Troubleshooting

- **No .txt file created?** Check if Tesseract is installed by running `tesseract --version`.
- **Text is garbled?** Ensure the image is clear. Tesseract works best with high-contrast text.
- **Script permission denied?** Run `chmod +x 0_meta/scripts/*.sh`.
