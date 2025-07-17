from microlia.training import run_training
from microlia.data_loading import load_lightcurve_data
import os

# paths to your combined dataset
base_dir = "/data01/aschweitzer/software/microlia_output/combined"
lc_file = os.path.join(base_dir, "microlia_combined_lightcurves.csv")
label_file = os.path.join(base_dir, "microlia_combined_labels.csv")

# Load data
X, y, meta = load_lightcurve_data(
    lightcurve_csv_path=lc_file,
    label_csv_path=label_file,
    label_column="label",
    id_column="id"
)

#output directory for models and logs
output_dir = os.path.join(base_dir, "trained_model")
os.makedirs(output_dir, exist_ok=True)

#now actually train model
run_training(
    X, y,
    output_dir=output_dir,
    n_trials=30,               # number of hyperparameter optimization trials
    time_budget=600,           # seconds for hyperparameter tuning
    test_size=0.2,             # train/test split ratio
    random_state=42,           # now for reproducibility
    verbosity=1
)
