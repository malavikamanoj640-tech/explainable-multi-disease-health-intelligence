from utils.pdf_reader import extract_text
from algorithms.report_parser import parse_report
from algorithms.report_type_detector import detect_report_type


text = extract_text("uploads/sample.pdf")

print("------------TEXT------------")

print(text)

print("\nReport Type")

print(detect_report_type(text))

print("\nExtracted Values")

print(parse_report(text))