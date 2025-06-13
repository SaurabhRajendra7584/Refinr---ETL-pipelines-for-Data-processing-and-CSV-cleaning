import pandas as pd
import json
import PyPDF2
from docx import Document

def parse_file(file_path, file_ext):
    if file_ext == 'csv':
        return pd.read_csv(file_path)
    
    elif file_ext in ['xls', 'xlsx']:
        return pd.read_excel(file_path)
    
    elif file_ext == 'json':
        return pd.read_json(file_path)
    
    elif file_ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        return pd.DataFrame({'text': [data]})
    
    elif file_ext == 'pdf':
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ''
        return pd.DataFrame({'text': [text]})
    
    elif file_ext == 'docx':
        text = ""
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + '\n'
        return pd.DataFrame({'text': [text]})
    
    else:
        raise Exception("Unsupported file format")
