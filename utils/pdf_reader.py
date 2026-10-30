import os
import pytesseract

from PIL import Image
from pypdf import PdfReader
from pdf2image import convert_from_path


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# NORMAL PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(filepath):

    text = ""

    try:

        reader = PdfReader(filepath)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception as e:

        print("Normal PDF extraction error:", e)

    return text.strip()


# ============================================================
# OCR EXTRACTION
# ============================================================

def extract_text_with_ocr(filepath):

    text = ""

    try:

        print("Starting OCR...")

        pages = convert_from_path(
            filepath,
            dpi=300
        )

        for page_number, page in enumerate(pages, start=1):

            print(f"OCR processing page {page_number}...")

            page_text = pytesseract.image_to_string(
                page,
                config="--psm 6"
            )

            if page_text:

                text += page_text + "\n"

        print("OCR completed.")

    except Exception as e:

        print("OCR ERROR:", e)

    return text.strip()


# ============================================================
# MAIN FUNCTION
# ============================================================

def extract_text(filepath):

    print("=" * 70)
    print("PDF TEXT EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Try normal PDF extraction
    # --------------------------------------------------------

    text = extract_text_from_pdf(filepath)

    print("Normal extraction characters:", len(text))

    # --------------------------------------------------------
    # STEP 2: If insufficient text, use OCR
    # --------------------------------------------------------

    if len(text.strip()) < 50:

        print()
        print("Little or no readable PDF text found.")
        print("Switching to OCR...")

        text = extract_text_with_ocr(filepath)

    else:

        print("Normal PDF text extraction successful.")

    print()
    print("Final extracted characters:", len(text))

    print("=" * 70)
    print("EXTRACTED TEXT")
    print("=" * 70)

    print(text)

    print("=" * 70)

    return text