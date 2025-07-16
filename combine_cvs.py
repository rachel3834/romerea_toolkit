import pandas as pd
import os

#path
base_dir = "/data01/aschweitzer/software/microlia_output"

#available classes
classes = ["const", "cv", "lpv", "ml", "rrlyr"]

#storage space for combined data
lightcurve_dfs = []
label_dfs = []

for cls in classes:
    print(f"Loading: {cls}")

    lc_path = os.path.join(base_dir, f"{cls}_microlia_lightcurves.csv")
    label_path = os.path.join(base_dir, f"{cls}_microlia_labels.csv")

    if not (os.path.exists(lc_path) and os.path.exists(label_path)):
        print(f"Missing files for class {cls}")
        continue

    #load + store
    lc_df = pd.read_csv(lc_path)
    label_df = pd.read_csv(label_path)

    lightcurve_dfs.append(lc_df)
    label_dfs.append(label_df)

#combine all
combined_lc = pd.concat(lightcurve_dfs, ignore_index=True)
combined_labels = pd.concat(label_dfs, ignore_index=True)

#output paths
output_dir = os.path.join(base_dir, "combined")
os.makedirs(output_dir, exist_ok=True)

lc_out = os.path.join(output_dir, "microlia_combined_lightcurves.csv")
label_out = os.path.join(output_dir, "microlia_combined_labels.csv")

combined_lc.to_csv(lc_out, index=False)
combined_labels.to_csv(label_out, index=False)

print(f"\n Lightcurves saved to {lc_out}")
print(f"Labels saved to {label_out}")
