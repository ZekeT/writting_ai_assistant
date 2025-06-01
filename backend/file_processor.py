import os
import tempfile
import subprocess
from pathlib import Path
# import docx2txt
import streamlit as st

# def convert_docx_to_markdown(file_content):
#     """
#     Convert DOCX file content to markdown.

#     Args:
#         file_content: Binary content of the DOCX file

#     Returns:
#         Markdown text
#     """
#     # Create a temporary file to store the DOCX content
#     with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
#         temp_file.write(file_content)
#         temp_file_path = temp_file.name

#     try:
#         # Use docx2txt to extract text
#         markdown_text = docx2txt.process(temp_file_path)

#         # Basic formatting improvements
#         # Add proper headers
#         lines = markdown_text.split('\n')
#         formatted_lines = []

#         for i, line in enumerate(lines):
#             # Check if this might be a header (short line followed by empty line)
#             if line.strip() and i < len(lines) - 1 and not lines[i + 1].strip():
#                 if len(line) < 100:  # Likely a header if relatively short
#                     if i == 0 or not lines[i - 1].strip():  # First line or preceded by empty line
#                         formatted_lines.append(f"# {line}")
#                         continue
#                     else:
#                         formatted_lines.append(f"## {line}")
#                         continue

#             # Regular line
#             formatted_lines.append(line)

#         markdown_text = '\n'.join(formatted_lines)

#         return markdown_text
#     finally:
#         # Clean up the temporary file
#         os.unlink(temp_file_path)


def convert_pdf_to_markdown(file_content):
    """
    Convert PDF file content to markdown using pdftotext from poppler-utils.

    Args:
        file_content: Binary content of the PDF file

    Returns:
        Markdown text
    """
    # Create a temporary file to store the PDF content
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf_file:
        temp_pdf_file.write(file_content)
        temp_pdf_path = temp_pdf_file.name

    # Create a temporary file for the text output
    temp_txt_path = temp_pdf_path + '.txt'

    try:
        # Use pdftotext (from poppler-utils) to extract text
        result = subprocess.run(
            ['pdftotext', '-layout', temp_pdf_path, temp_txt_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            st.error(f"Error converting PDF: {result.stderr}")
            return f"Error converting PDF: {result.stderr}"

        # Read the extracted text
        with open(temp_txt_path, 'r', encoding='utf-8') as txt_file:
            text_content = txt_file.read()

        # Basic formatting improvements
        lines = text_content.split('\n')
        formatted_lines = []

        for i, line in enumerate(lines):
            # Check if this might be a header (short line followed by empty line)
            if line.strip() and i < len(lines) - 1 and not lines[i + 1].strip():
                if len(line) < 100:  # Likely a header if relatively short
                    # First line or preceded by empty line
                    if i == 0 or not lines[i - 1].strip():
                        formatted_lines.append(f"# {line}")
                        continue
                    else:
                        formatted_lines.append(f"## {line}")
                        continue

            # Regular line
            formatted_lines.append(line)

        markdown_text = '\n'.join(formatted_lines)

        return markdown_text
    finally:
        # Clean up the temporary files
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
        if os.path.exists(temp_txt_path):
            os.unlink(temp_txt_path)


def extract_title_from_markdown(markdown_text, filename=""):
    """
    Extract a title from markdown content.

    Args:
        markdown_text: Markdown text
        filename: Original filename as fallback

    Returns:
        Extracted title
    """
    # Try to find a level 1 header
    lines = markdown_text.split('\n')
    for line in lines:
        if line.startswith('# '):
            return line.replace('# ', '').strip()

    # If no header found, use the first non-empty line
    for line in lines:
        if line.strip():
            # Limit title length
            title = line.strip()
            if len(title) > 60:
                title = title[:57] + '...'
            return title

    # If all else fails, use the filename without extension
    if filename:
        base_name = Path(filename).stem
        return base_name.replace('_', ' ').title()

    # Last resort
    return "Untitled Article"


def process_uploaded_file(uploaded_file):
    """
    Process an uploaded file and convert it to markdown.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Tuple of (title, markdown_content)
    """
    file_content = uploaded_file.read()
    file_extension = Path(uploaded_file.name).suffix.lower()

    # Convert based on file type
    if file_extension == '.docx':
        # markdown_text = convert_docx_to_markdown(file_content)
        raise NotImplementedError
    elif file_extension == '.pdf':
        markdown_text = convert_pdf_to_markdown(file_content)
    elif file_extension in ['.md', '.txt']:
        # Already text, just decode
        markdown_text = file_content.decode('utf-8')
    else:
        return None, f"Unsupported file type: {file_extension}"

    # Extract title
    title = extract_title_from_markdown(markdown_text, uploaded_file.name)

    return title, markdown_text
