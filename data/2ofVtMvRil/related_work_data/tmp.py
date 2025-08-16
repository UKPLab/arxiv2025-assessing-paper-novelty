import pikepdf
import os


def repair_pdf_with_pikepdf(input_pdf, output_pdf):
    """
    Opens and saves a PDF using pikepdf, which often fixes structure and removes problematic elements.
    """
    try:
        with pikepdf.open(input_pdf, allow_overwriting_input=True) as pdf:
            pdf.save(output_pdf)
        print(f"Sanitized PDF saved as: {output_pdf}")
    except Exception as e:
        print(f"Failed to repair PDF: {e}")


# Usage Example
pdfs = os.listdir("./pdfs")
for pdf in pdfs:
    print(f"Repairing {pdf}")
    repair_pdf_with_pikepdf(f"./pdfs/{pdf}", f"./pdfs/{pdf}")
