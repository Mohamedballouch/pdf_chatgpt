from PyPDF2 import PdfReader

def load_pdf_and_extract_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text
