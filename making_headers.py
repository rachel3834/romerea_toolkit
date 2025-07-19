import os

#training data organized by label (e.g., CV, CONST)
training_data_dir = "/data01/aschweitzer/software/microlia_output/training_data"
header = "time,mag,mag_err,filter\n"

#loop through subfolders
for label_dir in os.listdir(training_data_dir):
    label_path = os.path.join(training_data_dir, label_dir)
    if not os.path.isdir(label_path):
        continue

    for filename in os.listdir(label_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(label_path, filename)

            #read original contents
            with open(filepath, "r") as f:
                contents = f.read()

            #rewrite with header on top
            with open(filepath, "w") as f:
                f.write(header + contents)

print("All star csvs patched so that header was added at top of each file in order to ensure consistency before training.")
