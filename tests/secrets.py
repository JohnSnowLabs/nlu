import json
import os

license_dict = json.loads(os.getenv("JSL_LICENSE"))
AWS_ACCESS_KEY_ID = license_dict.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = license_dict.get("AWS_SECRET_ACCESS_KEY")
JSL_SECRET = license_dict.get("SECRET")
SPARK_NLP_LICENSE = license_dict.get("SPARK_NLP_LICENSE")
OCR_SECRET = license_dict.get("SPARK_OCR_SECRET")
OCR_LICENSE = license_dict.get("SPARK_OCR_LICENSE")
JSON_LIC_PATH = license_dict.get("JSON_LIC_PATH")
