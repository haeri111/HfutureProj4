import streamlit as st
import pdfplumber
from pypdf import PdfReader

# Page configuration
st.set_page_config(layout="wide", page_title="PDF 파일 업로드", page_icon="📄")

# Sidebar
st.sidebar.header("PDF 업로드 화면")

# File uploader widget
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

# File uploaded
if uploaded_file is not None:
    # Display file name
    st.write(f"업로드된 파일: {uploaded_file.name}")
    
    # Use pypdf to read the PDF metadata and number of pages
    reader = PdfReader(uploaded_file)
    num_pages = len(reader.pages)
    
    # Use pdfplumber to extract text from the first page
    with pdfplumber.open(uploaded_file) as pdf:
        first_page = pdf.pages[0].extract_text()
    
    # Display number of pages in the PDF
    st.write(f"이 PDF 파일에는 {num_pages} 페이지가 있습니다.")
    
    # Display text from the first page
    st.subheader("첫 번째 페이지 내용:")
    if first_page:
        st.text(first_page)
    else:
        st.write("첫 번째 페이지에서 텍스트를 추출할 수 없습니다.")

else:
    st.write("PDF 파일을 업로드하면 내용이 표시됩니다.")