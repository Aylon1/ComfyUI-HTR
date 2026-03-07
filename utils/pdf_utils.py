import os
try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

def convert_pdf_to_images(pdf_path, first_page=None, last_page=None):
    """
    Converts a PDF file to a list of PIL Images.
    If first_page and last_page are provided, converts only those pages.
    """
    if not HAS_PDF2IMAGE:
        raise ImportError(
            "The 'pdf2image' library is required to process PDF files. "
            "Please install it using 'pip install pdf2image' and ensure 'poppler' is installed on your system."
        )
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # pdf2image uses 1-based indexing
    kwargs = {}
    if first_page is not None:
        kwargs["first_page"] = first_page
    if last_page is not None:
        kwargs["last_page"] = last_page
        
    images = convert_from_path(pdf_path, **kwargs)
    return images
