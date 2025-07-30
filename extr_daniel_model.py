import zipfile
import os

zip_path = r"/Users/Alaina/desktop/Vanderbilt REU code/RomeRea/MicroLIA_ensemble_model-20250729T232530Z-1-001.zip"
extract_dir = r"/Users/Alaina/desktop/Vanderbilt REU code/RomeRea/daniel_microlia_model"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extracted to:", extract_dir)