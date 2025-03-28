import PyPDF2

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file, handling encryption, errors, and blank pages."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)

        # Check if the PDF is encrypted
        if reader.is_encrypted:
            print(f"🔒 Error: {uploaded_file.name} is encrypted. Cannot extract text.")
            return "Error: Encrypted PDF"

        text = "\n".join([page.extract_text() or "" for page in reader.pages])

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Debug: Print extracted text
        print(f"📄 Extracted text from {uploaded_file.name} (First 500 chars):\n{text[:500]}")

        return text if text.strip() else "No extractable text found."
    
    except Exception as e:
        print(f"❌ Error extracting text from {uploaded_file.name}: {e}")
        return "Error: Could not process PDF"
