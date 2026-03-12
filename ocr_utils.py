# ocr_utils.py
import cv2
import pytesseract
import re
from dateutil import parser
from datetime import datetime

def preprocess_for_ocr_bgr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray,9,75,75)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return th

def extract_text_from_bgr(img_bgr):
    th = preprocess_for_ocr_bgr(img_bgr)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(th, config=config)
    return text

def find_date_in_text(text):
    date_patterns = r'(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})|(\d{4}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{1,2})|(\d{1,2}[\/\-\.\s]\d{4})'
    matches = re.findall(date_patterns, text)
    candidates = [m for tup in matches for m in tup if m!='']
    for c in candidates:
        try:
            d = parser.parse(c, dayfirst=True, fuzzy=True)
            return d.date(), c
        except:
            continue
    return None, None