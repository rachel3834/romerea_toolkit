import os
from MicroLIA import training_set, ensemble_model
import pandas as pd

dat_dir = r"/Users/Alaina/desktop/Vanderbilt REU code/RomeRea/Training Data/microlia_output/training_data_i"
print(os.path.isdir(dat_dir))

filt = 'i'
trial_num = 1
csv_name = f"{filt}_trial{trial_num}"

# save CSV inside dat_dir
data_x, data_y = training_set.load_all(
    path=dat_dir,
    convert=True,
    save_file=True,
    zp=22,
    filename=csv_name
)
print("After load_all call")

# read CSV back from dat_dir
csv_full_path = "/Users/Alaina/MicroLIA_Training_Set_i_trial1.csv"
print(f"Looking for CSV at {csv_full_path}")


if os.path.exists(csv_full_path):
    print("CSV file exists!")
else:
    print("CSV file NOT found!")


csv = pd.read_csv(csv_full_path)

model = ensemble_model.Classifier(training_data=csv, clf='xgb', impute=True, optimize=True, n_iter=0, boruta_trials=100)
model.load('daniel_microlia_model/MicroLIA_ensenmble_model')

# plots
model.plot_conf_matrix(k_fold=10, savefig=True)
model.plot_tsne(norm=True, savefig=True)
model.plot_feature_opt(feat_names='default', top=10, include_other=True, include_shadow=True, include_rejected=False, flip_axes=True, savefig=True)
model.plot_feature_opt(feat_names='default', top=30, include_other=True, include_shadow=True, include_rejected=False, flip_axes=False, savefig=True)

model.save_hyper_importance()

model.plot_hyper_param_importance(plot_time=True, savefig=True)

model.save()
