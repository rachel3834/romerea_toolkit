from MicroLIA import training_set, ensemble_model
import os
import pandas as pd
import numpy as np


trial_num = 5
filt = 'g'

path = '/data01/aschweitzer/software/microlia_output/training_data_g/'
data_x, data_y = training_set.load_all(
    path=path,
    convert=True,
    zp=22,
    filename=f'ROME_{filt}_TRAINING_trial{trial_num}',
    apply_weights=True,
    save_file=True
)

#load in .txt to regenerate x and y data
from pathlib import Path
home = os.path.expanduser("~")
data = np.loadtxt(f'{home}/all_features_ROME_{filt}_TRAINING_trial{trial_num}.txt', dtype=str, comments='#')
data_x = data[:,2:].astype('float')
data_y = data[:,0]

#load in csv (not entirely necessary)
csv_path = os.path.join(home, f"MicroLIA_Training_Set_ROME_{filt}_TRAINING_trial{trial_num}.csv")
csv = pd.read_csv(csv_path)

model = ensemble_model.Classifier(data_x, data_y, clf='xgb', impute=True, optimize=True, n_iter=0, boruta_trials=80)
model.create()

#save location ~home
save_dir = f'test_model_{filt}_trial{trial_num}'
os.makedirs(save_dir, exist_ok=True)

#plots
model.plot_conf_matrix(k_fold=10, savefig=True)

model.plot_tsne(norm=True, savefig=True)

model.plot_feature_opt(feat_names='default', top=10, include_other=True, include_shadow=True, include_rejected=False, flip_axes=True, savefig=True)

model.plot_feature_opt(feat_names='default', top=30, include_other=True, include_shadow=True, include_rejected=False, flip_axes=False, savefig=True)

model.plot_hyper_opt(xlim=(1, 50), ylim=(0.92, 0.98), xlog=True, savefig=True)

model.save_hyper_importance(savefig=True)

model.plot_hyper_param_importance(plot_time=True, savefig=True)

#save to save_dir in ~home if possible
model.save(dirname=save_dir)