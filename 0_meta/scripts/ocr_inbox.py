#!/usr/bin/env python3
import os
import glob
from pathlib import Path

# Try to import pytesseract, but don't fail if not present
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

INBOX_DIR = "1_inbox"
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]

def run_ocr():
    if not HAS_OCR:
        print("Error: OCR libraries not found.")
        print("Please install them using: uv add pytesseract Pillow")
        print("And ensure tesseract is installed on your system: brew install tesseract")
        return

    inbox_path = Path(INBOX_DIR)
    if not inbox_path.exists():
        print(f"Inbox directory '{INBOX_DIR}' not found.")
        return

    for ext in IMAGE_EXTENSIONS:
        for img_path in glob.glob(str(inbox_path / f"*{ext}")):
            p = Path(img_path)
            txt_path = p.with_suffix(".txt")
            
            if txt_path.exists():
                print(f"Skipping {p.name} (TXT already exists)")
                continue
                
            print(f"Processing {p.name}...")
            try:
                text = pytesseract.image_to_string(Image.open(p))
                with open(txt_path, "w") as f:
                    f.write(text)
                print(f"  -> Created {txt_path.name}")
            except Exception as e:
                print(f"  -> Failed: {e}")

if __name__ == "__main__":
    run_ocr()
