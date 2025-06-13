import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'txt', 'pdf', 'docx'}
    SECRET_KEY = 'your_secret_key_here'
