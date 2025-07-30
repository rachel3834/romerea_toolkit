import tarfile
import os

tar_path = r"/Users/Alaina/desktop/Vanderbilt REU code/RomeRea/Training Data/microlia_output.tar.gz"
extract_dir = r"/Users/Alaina/desktop/Vanderbilt REU code/RomeRea/Training Data"


with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=extract_dir)


print("Done extracting to:", extract_dir)