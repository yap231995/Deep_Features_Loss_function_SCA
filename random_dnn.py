import argparse
import json
import sys

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras import *
import numpy as np

from src.config import Config
from src.hyperparameters import *
import importlib

from src.loss_fn import flr_loss
from src.net import Custom_Model
from src.training import train
# tf.config.run_functions_eagerly(True)

def get_reg(hp):
    if hp["regularization"] == "l1":
        return l1(l=hp["l1"])
    elif hp["regularization"] == "l2":
        return l2(l=hp["l2"])
    else:
        return hp["dropout"]

def mlp_random(classes, number_of_samples, loss_type,profiling_label = None, regularization=False, hp=None):
    # hp = get_hyperparameters_mlp(loss_type=loss_type, regularization=regularization) if hp is None else hp

    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    inputs = Input(shape=number_of_samples)

    x = None
    intermediate_layer = []
    for layer_index in range(hp["layers"]):
        if regularization and hp["regularization"] != "dropout":
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_regularizer=get_reg(hp),
                      kernel_initializer=hp["kernel_initializer"],
                      name='dense_{}'.format(layer_index))(inputs if layer_index == 0 else x)
        else:
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_initializer=hp["kernel_initializer"],
                      name='dense_{}'.format(layer_index))(inputs if layer_index == 0 else x)
        if regularization and hp["regularization"] == "dropout":
            x = Dropout(get_reg(hp))(x)
        intermediate_layer.append(x)
    outputs = Dense(classes, activation='softmax', name='predictions')(x)
    intermediate_layer.append(outputs)
    if loss_type == "soft_nn" or loss_type == "flr_soft_nn":
        model = Custom_Model([loss_type, hp["temperature"]], inputs, intermediate_layer) #TODO: temperature not yet implement thinking of optimize it to remove this hyperparameter.
    elif loss_type == "center_loss" or loss_type == "flr_center_loss":
        model = Custom_Model([loss_type, hp["alpha"], classes, intermediate_layer[-2].shape[1]], inputs,
                             intermediate_layer)
    else:
        model = Model(inputs, outputs)
    # optimizer = get_optimizer(hp["optimizer"], hp["learning_rate"])
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model, tf_random_seed, hp

def cnn_random(classes, number_of_samples,loss_type, profiling_label= None,regularization=False, hp = None):
    # hp = get_hyperparameters_mlp(loss_type=loss_type, regularization=regularization) if hp is None else hp
    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    inputs = Input(shape=(number_of_samples, 1))

    x = None
    intermediate_layer = []
    for layer_index in range(hp["conv_layers"]):
        x = Conv1D(kernel_size=hp["kernels"][layer_index], strides=hp["strides"][layer_index], filters=hp["filters"][layer_index],
                   activation=hp["activation"], padding="same")(inputs if layer_index == 0 else x)
        if hp["pooling_types"][layer_index] == "Average":
            x = AveragePooling1D(pool_size=hp["pooling_sizes"][layer_index], strides=hp["pooling_strides"][layer_index], padding="same")(x)
        else:
            x = MaxPooling1D(pool_size=hp["pooling_sizes"][layer_index], strides=hp["pooling_strides"][layer_index], padding="same")(x)
        x = BatchNormalization()(x)
        intermediate_layer.append(x)
    x = Flatten()(x)

    for layer_index in range(hp["layers"]):
        if regularization and hp["regularization"] != "dropout":
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_regularizer=get_reg(hp),
                      kernel_initializer=hp["kernel_initializer"], name='dense_{}'.format(layer_index))(x)
        else:
            x = Dense(hp["neurons"], activation=hp["activation"], kernel_initializer=hp["kernel_initializer"],
                      name='dense_{}'.format(layer_index))(x)
        if regularization and hp["regularization"] == "dropout":
            x = Dropout(get_reg(hp))(x)

        intermediate_layer.append(x)
    outputs = Dense(classes, activation='softmax', name='predictions')(x)
    intermediate_layer.append(outputs)
    if loss_type == "soft_nn" or loss_type == "flr_soft_nn":
        print("temperature:", hp["temperature"])
        model = Custom_Model([loss_type, hp["temperature"]], inputs, intermediate_layer) #TODO: temperature not yet implement thinking of optimize it to remove this hyperparameter.
    elif loss_type == "center_loss" or loss_type == "flr_center_loss":
        model = Custom_Model([loss_type, hp["alpha"], classes,intermediate_layer[-2].shape[1]], inputs, intermediate_layer)
    else:
        model = Model(inputs, outputs)

    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model, tf_random_seed, hp


def get_optimizer(optimizer, learning_rate):
    module_name = importlib.import_module("tensorflow.keras.optimizers")
    optimizer_class = getattr(module_name, optimizer)
    return optimizer_class(lr=learning_rate)



if __name__ == "__main__":

    from src.utils import load_aes_hd_ext, load_chipwhisperer_desync, load_chipwhisperer, load_ascad, \
    attack_calculate_metrics, load_aes_rd, split_profiling, load_ctf, str2bool, load_aes_hd_ext_id, NTGE_fn
    from sklearn.preprocessing import StandardScaler
    import os
    import time
    from tensorflow.keras.utils import to_categorical
    # from src.sca_metrics import sca_metrics
    root = './'


    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_id", default=config.general.search_id, type=int)
    parser.add_argument("--dataset", default=config.general.dataset)
    parser.add_argument("--nb_traces_attacks", default=config.general.nb_traces_attacks, type=int)
    parser.add_argument("--leakage_model", default=config.general.leakage_model)
    parser.add_argument("--model_type", default=config.general.model_type)
    parser.add_argument("--loss_type", default=config.general.loss_type)
    parser.add_argument("--regularization", default=config.general.regularization, type = str2bool, choices=[True,False])
    parser.add_argument("--epochs", default=config.general.epochs, type = int)
    # parser.add_argument("--hp", default=config.general.hp, type=json.loads)
    args = parser.parse_args()
    dataset = args.dataset  # ASCAD #ASCAD_variable #AES_HD_ext #AES_HD_ext_ID #CTF #ASCADv2
    nb_traces_attacks = args.nb_traces_attacks
    model_type = args.model_type
    leakage = args.leakage_model
    loss_type = args.loss_type
    search_id = args.search_id
    regularization = args.regularization
    epochs = args.epochs
    print("search_id", search_id)
    print("epochs", epochs)
    save_root = root + 'Result/' + dataset + '_' + leakage +'_epochs_'+str(epochs)+'/'
    search_root = save_root + model_type + '/'
    loss_root = search_root + loss_type + '/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    print("search_root: ", search_root)
    print("loss_root: ", loss_root)
    if not os.path.exists(search_root):
        os.mkdir(search_root)
    if not os.path.exists(loss_root):
        os.mkdir(loss_root)

    hp = np.load(search_root + '/hp_'+model_type+'_'+str(search_id)+".npy", allow_pickle = True).item()
    print("hp inside: ", hp)
    # print(type(hp))


    # byte = 2
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

    elif dataset == 'ASCAD_k0':
        byte = 0
        data_root = 'Dataset/ASCAD/ASCAD_k0.h5'
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
    elif dataset == 'Chipwhisperer':
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (
        plt_profiling, plt_attack), correct_key = load_chipwhisperer(
            root + data_root + '/', leakage_model=leakage)
    elif dataset == 'Chipwhisperer_desync25':
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_chipwhisperer_desync(root + data_root + '/', desync_lvl=25,
                                                                                leakage_model=leakage)
    elif dataset == 'Chipwhisperer_desync50':
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_chipwhisperer_desync(root + data_root + '/', desync_lvl=50,
                                                                                leakage_model=leakage)


    elif dataset == 'AES_HD_ext':
        data_root = 'Dataset/AES_HD_ext/aes_hd_ext.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_aes_hd_ext(
            root + data_root, leakage_model=leakage, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)
    elif dataset == "AES_HD_ext_ID":
        data_root = 'Dataset/AES_HD_ext/aes_hd_ext.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_aes_hd_ext_id(
            root + data_root, leakage_model=leakage, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)

    elif dataset == 'AES_RD':
        data_root = 'Dataset/AES_RD/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_aes_rd(root +data_root,leakage_model=leakage, train_begin=0, train_end=20000, test_begin=0,
            test_end=nb_traces_attacks)


    print("The dataset we using: ", data_root)
    # load the data and normalize it

    print("Number of X_attack used: ", X_attack.shape)
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
    # loss_type = 'categorical_crossentropy'  # soft_nn #center_loss #categorical_crossentropy


    # number_of_searches = 100


    nb_attacks = 100

    # good_model_search_id_lst = []
    # for search_id in range(number_of_searches):
    tf.keras.backend.clear_session()
    start_time = time.time()

    """ Create random model """
    hp["epochs"] = epochs

    if model_type == "mlp":
        model, seed, hp = mlp_random(classes, number_of_samples, loss_type,profiling_label=y_train, regularization=regularization, hp=hp)
    else:
        model, seed, hp = cnn_random(classes, number_of_samples, loss_type,profiling_label=y_train,  regularization=regularization,hp = hp)

    """ Train model """
    ##BASE LOSS
    if loss_type == 'categorical_crossentropy' or loss_type == 'center_loss' or loss_type == 'soft_nn':
        base_loss_fn = 'categorical_crossentropy'
        base_loss_fn_actual = "categorical_crossentropy"
    elif loss_type == 'flr' or loss_type == "flr_center_loss" or loss_type == "flr_soft_nn":
        base_loss_fn = 'flr'
        base_loss_fn_actual = flr_loss(10)


    if loss_type == "center_loss" or loss_type == "soft_nn" or loss_type == "flr_soft_nn" or loss_type == "flr_center_loss":
        lamb_str = "lamb"
        if loss_type == "soft_nn" or loss_type == "flr_soft_nn":
            lamb_str = "lamb_softnn"
        history =train(x_train, to_categorical(y_train, num_classes=classes), x_val, to_categorical(y_val, num_classes=classes), model, epochs, hp["batch_size"], hp[lamb_str], optimizer = get_optimizer(hp["optimizer"], hp["learning_rate"]), base_loss_fn = base_loss_fn)
        new_model = Model([model.input], [model.output[-1]])
    else:
        model.compile(loss=base_loss_fn_actual,optimizer = get_optimizer(hp["optimizer"], hp["learning_rate"]), metrics=['accuracy'])
        history = model.fit(x=x_train, y=to_categorical(y_train, num_classes=classes), batch_size=hp["batch_size"], verbose=2,
                  epochs=epochs, validation_data=(x_val, to_categorical(y_val, num_classes=classes)))
        new_model = model
    """ Compute GE, SR and NT for attack set """
    end_time = time.time() - start_time
    attack_phase = True
    if attack_phase == True:
        container = attack_calculate_metrics(new_model,X_attack=X_attack,Y_attack = Y_attack,plt_attack=plt_attack,leakage = leakage, nb_attacks=nb_attacks,nb_traces_attacks = nb_traces_attacks,correct_key=correct_key,dataset =dataset)
        GE = container[257:]
        print("GE:", GE)
        NTGE = NTGE_fn(GE)
        print("NTGE:", NTGE)
    else:
        NTGE = None

    np.save(loss_root + '/misc_{}_{}_{}_{}'.format(dataset, leakage, model_type, search_id),
            {"seed": seed, "elapse_time": end_time, "NTGE": NTGE})
    np.save(loss_root + '/Result_{}_{}_{}_{}'.format(dataset, leakage, model_type, search_id), container)
    if GE[-1] < 1:
        print("SAVING GOOD CNN")
        np.save(loss_root + '/hyperparameters_{}_{}_{}_{}.npy'.format(dataset, leakage,model_type,search_id), hp)
        np.save(loss_root + '/history_{}_{}_{}_{}.npy'.format(dataset, leakage,model_type,search_id), history)
        new_model.save(loss_root + dataset + "_" + leakage+ "_" + model_type + "_"+str(search_id))

