import json
import os

if os.path.exists('./tests/lic.json'):
    with open('./tests/lic.json', 'r') as f:
        license_dict = json.loads(f.read())
elif 'JOHNSNOWLABS_LICENSE_JSON' in os.environ:
    license_dict = json.loads(os.getenv("JOHNSNOWLABS_LICENSE_JSON"))
    with open('./tests/lic.json', 'w') as f:
        json.dump(license_dict, f)
else:
    raise Exception("No license found")

AWS_ACCESS_KEY_ID = license_dict.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = license_dict.get("AWS_SECRET_ACCESS_KEY")
JSL_SECRET = license_dict.get("SECRET")
SPARK_NLP_LICENSE = license_dict.get("SPARK_NLP_LICENSE")
OCR_SECRET = license_dict.get("SPARK_OCR_SECRET")
OCR_LICENSE = license_dict.get("SPARK_OCR_LICENSE")

JSON_LIC_PATH = './tests/lic.json'
