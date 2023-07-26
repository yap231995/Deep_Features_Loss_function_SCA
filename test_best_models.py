import argparse
import gc
import os
import sys

import tensorflow as tf
from tensorflow.keras.models import *
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from random_dnn import mlp_random, cnn_random, get_optimizer
from src.config import Config
from src.loss_fn import flr_loss
from src.training import train
from src.utils import load_ascad, load_ctf, load_chipwhisperer, load_chipwhisperer_desync, load_aes_hd_ext, load_aes_rd, \
    load_aes_hd_ext_id, split_profiling, attack_calculate_metrics, NTGE_fn, str2bool


perf_att = True

config = Config()
parser = argparse.ArgumentParser()
parser.add_argument("--best_model_idx", default=config.general.best_model_idx, type=int)
parser.add_argument("--num_training_index", default=config.general.num_training_index, type=int)
parser.add_argument("--dataset", default=config.general.dataset)
parser.add_argument("--nb_traces_attacks", default=config.general.nb_traces_attacks, type=int)
parser.add_argument("--leakage_model", default=config.general.leakage_model)
parser.add_argument("--model_type", default=config.general.model_type)
parser.add_argument("--loss_type", default=config.general.loss_type)
parser.add_argument("--regularization", default=config.general.regularization, type=str2bool, choices=[True, False])
parser.add_argument("--epochs", default=config.general.regularization, type=int)
args = parser.parse_args()
dataset = args.dataset  # ASCAD #ASCAD_variable #AES_HD_ext #AES_HD_ext_ID #CTF #ASCADv2
nb_traces_attacks = args.nb_traces_attacks
model_type = args.model_type
leakage = args.leakage_model
loss_type = args.loss_type
best_model_idx = args.best_model_idx
num_training_index = args.num_training_index
regularization = args.regularization
epochs = args.epochs
print("best_model_idx", best_model_idx)
print("num_training_index", num_training_index)
root = './'
save_root = root + 'Result/' + dataset + '_' + leakage + '_epochs_'+str(epochs)+ '/'
search_root = save_root + model_type + '/'
loss_root = search_root + loss_type + '/'
best_root = loss_root + 'best_models/'
if not os.path.exists(save_root):
    os.mkdir(save_root)
print("search_root: ", search_root)
print("loss_root: ", loss_root)
if not os.path.exists(search_root):
    os.mkdir(search_root)
if not os.path.exists(loss_root):
    os.mkdir(loss_root)
if not os.path.exists(best_root):
    os.mkdir(best_root)

if dataset == 'ASCAD':
    byte = 2
    data_root = 'Dataset/ASCAD/ASCAD.h5'
    (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
        root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=50000, test_begin=0,
        test_end=nb_traces_attacks)
elif dataset == 'ASCAD_desync50':
    byte = 2
    data_root = 'Dataset/ASCAD/ASCAD_desync50.h5'
    (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
        root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=50000, test_begin=0,
        test_end=nb_traces_attacks)
elif dataset == 'ASCAD_desync100':
    byte = 2
    data_root = 'Dataset/ASCAD/ASCAD_desync100.h5'
    (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
        root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=50000, test_begin=0,
        test_end=nb_traces_attacks)

elif dataset == 'ASCAD_variable':
    byte = 2
    data_root = 'Dataset/ASCAD/ASCAD_variable.h5'
    (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
        root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=50000, test_begin=0,
        test_end=nb_traces_attacks)
elif dataset == 'ASCAD_variable_desync50':
    byte = 2
    data_root = 'Dataset/ASCAD/ASCAD_variable_desync50.h5'
    (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
        root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
        test_end=nb_traces_attacks)

elif dataset == 'ASCAD_variable_desync100':
    byte = 2
    data_root = 'Dataset/ASCAD/ASCAD_variable_desync100.h5'
    (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
        root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
        test_end=nb_traces_attacks)
elif dataset == 'CTF2018':
    byte = 0
    data_root = 'Dataset/CTF2018/ches_ctf.h5'
    (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ctf(
        root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
        test_end=nb_traces_attacks)


if leakage == 'HW':
    classes = 9
    print("Number of Y_attack used: ", len(Y_attack))
elif leakage == 'ID':
    classes = 256
    print("Number of Y_attack used: ", Y_attack.shape)

input_length = len(X_profiling[0])
print('Input length: {}'.format(input_length))
scaler = StandardScaler()
X_profiling = scaler.fit_transform(X_profiling)
X_attack = scaler.transform(X_attack)

number_of_samples = X_profiling.shape[1]
X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1)).astype('float32')
X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1)).astype('float32')
x_train, x_val, y_train, y_val, plt_train, plt_val, keys_train, keys_val = split_profiling(X_profiling, Y_profiling,
                                                                                           plt_profiling,
                                                                                           correct_key, dataset,
                                                                                           percentage_val=0.1)


nb_attacks = 100
regularization = False
print("Loading HP from: ", loss_root + '/hyperparameters_{}_{}_{}_{}.npy'.format(dataset, leakage,model_type,best_model_idx))
hp = np.load(loss_root + '/hyperparameters_{}_{}_{}_{}.npy'.format(dataset, leakage,model_type,best_model_idx), allow_pickle=True).item()
hp["epochs"] = epochs
if model_type == "mlp":
    model, seed, hp = mlp_random(classes, number_of_samples, loss_type, regularization=regularization, hp=hp)
else:
    model, seed, hp = cnn_random(classes, number_of_samples, loss_type, regularization=regularization, hp=hp)

""" Train model """

##BASE LOSS
if loss_type == 'categorical_crossentropy' or loss_type == 'center_loss' or loss_type == 'soft_nn':
    base_loss_fn = 'categorical_crossentropy'
    base_loss_fn_actual = "categorical_crossentropy"
elif loss_type == 'flr' or loss_type == "flr_center_loss" or loss_type == "flr_soft_nn" :
    base_loss_fn = 'flr'
    base_loss_fn_actual = flr_loss(10)
if loss_type == "center_loss" or loss_type == "soft_nn" or loss_type == "flr_soft_nn" or loss_type == "flr_center_loss":
    lamb_str = "lamb"
    if loss_type == "soft_nn" or loss_type == "flr_soft_nn":
        lamb_str = "lamb_softnn"
    history = train(x_train, to_categorical(y_train, num_classes=classes), x_val,
                    to_categorical(y_val, num_classes=classes), model, epochs, hp["batch_size"], hp[lamb_str],
                    optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"]), base_loss_fn=base_loss_fn)
    model = Model([model.input], [model.output[-1]])
else:
    model.compile(loss=base_loss_fn_actual, optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"]),
                  metrics=['accuracy'])
    history = model.fit(x=x_train, y=to_categorical(y_train, num_classes=classes), batch_size=hp["batch_size"],
                        verbose=2,
                        epochs=epochs, validation_data=(x_val, to_categorical(y_val, num_classes=classes)))
    # new_model = model

# model.save(best_root + "best_model_" + str(num_training_index))
# model = load_model(best_root + "best_model_" + str(num_training_index) , compile=False)

if perf_att == True:
    print("num_training_index:", num_training_index)
    container = attack_calculate_metrics(model, X_attack=X_attack, Y_attack=Y_attack, plt_attack=plt_attack,
                                         leakage=leakage, nb_attacks=nb_attacks, nb_traces_attacks=nb_traces_attacks,
                                         correct_key=correct_key, dataset=dataset)
    GE = container[257:]

    NTGE = NTGE_fn(GE)
    print("GE:", GE)
    print("NTGE:", NTGE)
    np.save(best_root + 'GE_' + str(num_training_index), GE)


del model
tf.keras.backend.clear_session()
gc.collect()
sys.exit()