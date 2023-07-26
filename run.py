import os
from time import sleep
import numpy as np
import random
from src.hyperparameters import get_hyperparameters_mlp, get_hyperparemeters_cnn



number_of_searches = 100
regularization = False
epochs = 50
dataset = "ASCAD" # ASCAD #ASCAD_variable #AES_HD_ext #AES_HD_ext_ID #CTF2018 #ASCAD_desync50 #ASCAD_variable_desync50
nb_traces_attacks = 2000 #ASCAD:2000 ASCAD_var: 10000 CTF2018: 3000
for leakage in ["HW","ID"]:
    for model_type in ["mlp","cnn"]:
        for i in range(number_of_searches):
            print("search_id:", i)
            if model_type == "mlp":
                hp = get_hyperparameters_mlp(regularization=regularization)
            else:
                hp = get_hyperparemeters_cnn(regularization=regularization)
            # add the lamb_max_softnn
            hp['lamb_maxsoftnn'] = random.choice([ -1, -0.5, -0.1, -0.05, -0.01, -0.005, -0.001])
            # hp['alpha'] = 0.005
            # hp['lamb'] = 1
            # np.save(root, hp)
            save_root = './Result/' + dataset + '_' + leakage +'_epochs_'+str(epochs)+ '/'
            search_root = save_root + model_type + '/'
            if not os.path.exists(save_root):
                os.mkdir(save_root)
            if not os.path.exists(search_root):
                os.mkdir(search_root)
            root = search_root + '/hp_'+model_type+'_'+str(i)+".npy"
            np.save(root, hp)
            hp = np.load(root, allow_pickle=True).item()
            print("hp outside: ", hp)
            print("root outside: ", root)


            #To brute force to start at some place else.
            # if leakage == "HW" and model_type =="mlp":
            #     continue
            # elif leakage == "HW" and model_type == "cnn":
            #     if i < 43:
            #         continue
            #
            for loss_type in [ "soft_nn"]: #flr_weighted_soft_nn, flr_weighted_center_loss, 'soft_nn',#'categorical_crossentropy'#'center_loss' #flr #flr_soft_nn #flr_center_loss #max_soft_nn #flr_max_soft_nn
                os.system("python random_dnn.py --search_id "+str(i)+" --model_type "+model_type+" --leakage_model "+leakage + " --dataset " + dataset +
                          " --loss_type "+ loss_type + " --nb_traces_attacks " + str(nb_traces_attacks)+ " --regularization " + str(regularization)+" --epochs " +str(epochs))
                sleep(1)



