import pandas as pd
import os

#base directory
base_dir = "/data01/aschweitzer/software/microlia_output"

#map each class to its subdirectory
class_dirs = {
    "CONST": "const",
    "CV": "cv",
    "LPV": "lpv",
    "ML": "ml",
    "RRLYR": "rrlyr"
}

lightcurve_dfs = []
label_dfs = []

for label, subdir in class_dirs.items():
    subpath = os.path.join(base_dir, subdir)
    lc_file = os.path.join(subpath, f"{subdir}_microlia_lightcurves.csv")
    label_file = os.path.join(subpath, f"{subdir}_microlia_labels.csv")

    if not os.path.exists(lc_file) or not os.path.exists(label_file):
        print(f"Missing files for {label} in {subpath}")
        continue

    #load and patch label name if needed
    lc_df = pd.read_csv(lc_file)
    label_df = pd.read_csv(label_file)
    label_df["label"] = label  #ensuring this has the correct label name

    lightcurve_dfs.append(lc_df)
    label_dfs.append(label_df)

#combine everything
combined_lc = pd.concat(lightcurve_dfs, ignore_index=True)
combined_labels = pd.concat(label_dfs, ignore_index=True)

#output
output_dir = os.path.join(base_dir, "combined")
os.makedirs(output_dir, exist_ok=True)

lc_out = os.path.join(output_dir, "microlia_combined_lightcurves.csv")
label_out = os.path.join(output_dir, "microlia_combined_labels.csv")

combined_lc.to_csv(lc_out, index=False)
combined_labels.to_csv(label_out, index=False)

print(f"\nCombined lightcurves saved to {lc_out}")
print(f"Combined labels saved to {label_out}")
