import gdown
import os
import zipfile

url = 'https://drive.google.com/uc?id=16pe2VphW7n2gZXrCBjDXfUL1AbjOm2rw'
extract_to = 'docs_and_results/checkpoints/'
output_zip = 'downloaded_file.zip'

os.makedirs(os.path.expanduser("~/.cache/gdown"), exist_ok=True)
gdown.download(url, output=output_zip, quiet=False)

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

os.remove(output_zip)

print(f"✅ File downloaded and extracted to: {extract_to}")