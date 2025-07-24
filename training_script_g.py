from MicroLIA import training_set, ensemble_model
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

filter_used = "g"
base_training_path = "/data01/aschweitzer/software/microlia_output"

training_data_path = os.path.join(base_training_path, f"training_data_{filter_used}")
model_dir_name = os.path.join(base_training_path, f"model_{filter_used}")
csv_path = os.path.join(base_training_path, f"MicroLIA_Training_Set_{filter_used}.csv")

#load, validate data
data_x, data_y = training_set.load_all(path=training_data_path)
valid_rows = []

for arr, label in zip(data_x, data_y):
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 3:
        print(f"[SKIPPING!] Invalid shape {arr.shape}, expected (*,3)")
        continue
    df_single = pd.DataFrame(arr, columns=["time", "mag", "mag_err"])
    df_single["label"] = label
    valid_rows.append(df_single)

if not valid_rows:
    raise ValueError("No valid lightcurves found... check .dat file formatting.")

# concatenate and save
df = pd.concat(valid_rows, ignore_index=True)
df.to_csv(csv_path, index=False)
print(f"Saved training data CSV to: {csv_path}")
print(f"Loaded {len(data_x)} lightcurves for filter {filter_used}")

#train, save
model = ensemble_model.Classifier(
    data_x,
    data_y,
    impute=True,
    optimize=True,
    opt_cv=3,
    boruta_trials=25,
    n_iter=25
)
model.create()
model.save(dirname=model_dir_name)
print(f"Model saved to: {model_dir_name}")

#reload model to eval for cm
model_loaded = ensemble_model.Classifier(
    data_x,
    data_y,
    clf="xgb",
    impute=True
)
model_loaded.load(path=model_dir_name)
print("[INFO] Model reloaded from disk.")

# eval
y_pred = model_loaded.predict(data_x)
y_true = data_y

labels = sorted(set(y_true))
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

#conf matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title(f"Confusion Matrix - Filter {filter_used}")
plt.tight_layout()

final_path = os.path.join(base_training_path, "confusion_matrices")
os.makedirs(final_path, exist_ok=True)
plt.savefig(os.path.join(final_path, f"confusion_matrix_{filter_used}.png"))
plt.show()

print(f"Saved confusion matrix to: {final_path}")
