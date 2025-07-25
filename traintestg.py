from MicroLIA import training_set, ensemble_model

import os

trial_num = 1
filt = 'g'

path = '/data01/aschweitzer/software/microlia_output/training_data_g/'
data_x, data_y = training_set.load_all(
    path=path,
    convert=True,
    zp=22,
    filename='ROME_G_TRAINING',
    apply_weights=True,
    save_file=True
)

#--------- FIX --------------#
import numpy as np

#map original classes to sequential ones (0–3) because class consistency error
class_map = {0: 0, 2: 1, 3: 2, 4: 3}
data_y = np.array([class_map[y] for y in data_y])

#--------- FIX --------------#

#train
model = ensemble_model.Classifier(
    data_x, data_y,
    clf='xgb',
    impute=True,
    optimize=True,
    opt_cv=10,
    n_iter=0,
    boruta_trials=100
)


os.makedirs(f'microlia_output/trial{trial_num}', exist_ok=True)

model.create()
model.save(f'microlia_output/trial{trial_num}/ROME_{filt}_MODEL_{trial_num}')


import matplotlib.pyplot as plt

model.plot_conf_matrix()
plt.savefig(f'microlia_output/trial{trial_num}/conf_matrix_{filt}.png', bbox_inches='tight')
plt.close()

model.plot_tsne()
plt.savefig(f'microlia_output/trial{trial_num}/tsne_{filt}.png', bbox_inches='tight')
plt.close()

model.plot_feature_opt(top=20, flip_axes=True)
plt.savefig(f'microlia_output/trial{trial_num}/feature_opt_{filt}.png', bbox_inches='tight')
plt.close()

model.plot_hyper_opt(xlim=(1,100), ylim=(0.9775,0.995), xlog=True)
plt.savefig(f'microlia_output/trial{trial_num}/hyper_opt_{filt}.png', bbox_inches='tight')
plt.close()

model.plot_hyper_param_importance(plot_time=True)
plt.savefig(f'microlia_output/trial{trial_num}/hyper_param_importance_{filt}.png', bbox_inches='tight')
plt.close()

