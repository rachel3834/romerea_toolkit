from MicroLIA import training_set, ensemble_model
import os

trial_num = 1

path = '/data01/aschweitzer/software/microlia_output/training_data_r/'
data_x, data_y = training_set.load_all(
    path=path,
    convert=True,
    zp=22,
    filename='ROME_R_TRAINING',
    apply_weights=True,
    save_file=True
)

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

#create, save
model.create()
model.save('ROME_R_MODEL')


os.makedirs(f'microlia_output/{trial_num}', exist_ok=True)

#make plot
model.plot_conf_matrix(save_path='microlia_output/trial1/conf_matrix_r.png')                 #conf matrix
model.plot_tsne(save_path='microlia_output/trial1/tsne_r.png')                        #feature space projection
model.plot_feature_opt(top=20, flip_axes=True, save_path='microlia_output/trial1/plot_feature_opt_r.png')
model.plot_hyper_opt(xlim=(1,100), ylim=(0.9775,0.995), xlog=True, save_path='microlia_output/trial1/hyper_opt_r.png')
model.save_hyper_importance()
model.plot_hyper_param_importance(plot_time=True, save_path='microlia_output/trial1/hyper_param_importance_r.png')
