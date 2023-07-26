import gc
import os
import numpy as np
from src.utils import NTGE_fn
from tensorflow.keras.models import *
# from keras.utils import to_categorical
# from sklearn.preprocessing import StandardScaler
#
# from random_dnn import mlp_random, cnn_random, get_optimizer
# from src.loss_fn import cer_loss
# from src.training import train
# from src.utils import load_ascad, load_ctf, load_chipwhisperer, load_chipwhisperer_desync, load_aes_hd_ext, load_aes_rd, \
#     load_aes_hd_ext_id, split_profiling, attack_calculate_metrics, NTGE_fn

dataset = "ASCAD" # ASCAD #ASCAD_variable #AES_HD_ext #AES_HD_ext_ID #CTF2018 #ASCADv2
nb_traces_attacks = 10000 #ASCAD:2000 ASCAD_var: 10000 CTF2018: 3000
# leakage = 'ID'
# model_type = 'mlp'
# loss_type = 'categorical_crossentropy' #categorical_crossentropy #soft_nn
epochs = 50
for leakage in ['HW','ID']: #'HW'
    for model_type in ['mlp','cnn']:
        for loss_type in ['flr_center_loss']: #'max_soft_nn','soft_nn', 'categorical_crossentropy' center_loss #'flr' 'flr_soft_nn', 'flr_center_loss'
            number_of_searches = 100

            root = './'
            save_root = root + 'Result/' + dataset + '_' + leakage +'_epochs_'+str(epochs)+ '/'
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
            #1. Get this lst of successful models:
            search_successful_model = True
            cal_best_among_best_NTGE = True
            min_GE = float('inf')
            if search_successful_model == True:
                lst_successful_model = []
                min_GE = 256
                for search_id in range(number_of_searches):
                    if not os.path.exists(loss_root + '/misc_{}_{}_{}_{}.npy'.format(dataset, leakage, model_type, search_id)):
                        print(search_id)
                        continue
                    misc = np.load(loss_root + '/misc_{}_{}_{}_{}.npy'.format(dataset, leakage, model_type, search_id), allow_pickle= True).item()

                    # print("search_id:", search_id)
                    # print(misc)
                    NTGE =misc["NTGE"]
                    if NTGE != float('inf'):
                        lst_successful_model.append(search_id)

                    GE = np.load(loss_root + '/Result_{}_{}_{}_{}.npy'.format(dataset, leakage, model_type, search_id),
                                 allow_pickle=True)
                    # print("GE shape:", GE.shape)
                    if GE[-1] < min_GE:
                        min_GE = GE[-1]
                np.save(loss_root + '/lst_successful_model.npy', lst_successful_model)
                np.save(loss_root + '/min_GE.npy', min_GE)
                print("min_GE: ", min_GE)
            # print(ok)
            #2. Find the best model
            if not os.path.exists(loss_root+'/lst_successful_model.npy'):
                min_GE = np.load(loss_root + '/min_GE.npy', allow_pickle=True )
                print("min_GE: ", min_GE)
                print("dont have any model with GE = 0 ")
                continue
            lst_successful_model = np.load(loss_root+'/lst_successful_model.npy', allow_pickle= True)
            # print(lst_successful_model)
            best_NTGE = float('inf')
            best_model_idx = float('inf')
            for search_id in lst_successful_model:

                container = np.load(loss_root + '/Result_{}_{}_{}_{}.npy'.format(dataset, leakage, model_type, search_id),
                               allow_pickle=True)
                GE = container[257:]

                # print(misc)
                NTGE = NTGE_fn(GE)
                # print("NTGE:",NTGE)
                if NTGE < best_NTGE:
                    best_model_idx =search_id
                    best_NTGE = NTGE
            print("best_model_idx:", best_model_idx)
            print("best_NTGE:", best_NTGE)
            gc.collect()
            if os.path.exists(loss_root + '/hyperparameters_{}_{}_{}_{}.npy'.format(dataset, leakage, model_type, best_model_idx)):
                hp = np.load(
                    loss_root + '/hyperparameters_{}_{}_{}_{}.npy'.format(dataset, leakage, model_type, best_model_idx),
                    allow_pickle=True).item()
                print("best temperature:", hp["temperature"])
            else:
                min_GE = np.load(loss_root + '/min_GE.npy', allow_pickle=True )
                print("min_GE: ", min_GE)
                print("dont have path:", loss_root + '/hyperparameters_{}_{}_{}_{}.npy'.format(dataset, leakage, model_type, best_model_idx))
                continue
            #3. Load the hyperparameter and build the model, train this model for a few times and obtain the best results.
            nb_attacks = 100
            regularization = False
            # print("Loading HP from: ", loss_root + '/hyperparameters_{}_{}_{}_{}.npy'.format(dataset, leakage,model_type,best_model_idx))
            # model = load_model(loss_root + dataset + "_" + leakage + "_" + model_type + "_" + str(best_model_idx),
            #                    compile=False)
            # model.summary()
            # print(ok)
            # hp = np.load(loss_root + '/hyperparameters_{}_{}_{}_{}.npy'.format(dataset, leakage,model_type,best_model_idx), allow_pickle=True).item()
            # hp["epochs"] = epochs

            #Train a few times
            num_trainning = 10
            NTGE_best_among_best = float('inf')
            NTGE_lst = []
            trainning_best = True

            if cal_best_among_best_NTGE == True:
                # for num_training_index in range(num_trainning):
                num_training_index = 0
                while num_training_index < num_trainning:
                    print("num_training_index: {}/{}".format(num_training_index,num_trainning))
                    if trainning_best == True:
                        #Set os.system to run test_best_models.py
                        print("Start to train:")
                        os.system("python3 test_best_models.py --num_training_index "+str(num_training_index)+" --best_model_idx "+str(best_model_idx)+" --model_type "+model_type+" --leakage_model "+leakage + " --dataset " + dataset + " --loss_type "+ loss_type + " --nb_traces_attacks " + str(nb_traces_attacks)+ " --regularization " + str(regularization) + " --epochs " + str(epochs))
                    flag_exist_fails = False
                    if not os.path.exists(best_root + 'GE_' + str(num_training_index)+".npy"):
                        flag_exist_fails =True
                        continue
                    GE = np.load(best_root + 'GE_' + str(num_training_index)+".npy")
                    # print("GE.shape for training:", GE.shape)
                    NTGE = NTGE_fn(GE)
                    if NTGE < NTGE_best_among_best:
                        NTGE_best_among_best = NTGE
                    NTGE_lst.append(NTGE)
                    num_training_index += 1
                sorted_NTGE = np.sort(np.array(NTGE_lst))
                print("sorted_NTGE:", sorted_NTGE)
                if flag_exist_fails == False:
                    if num_trainning % 2 == 0:
                        median_NTGE = (sorted_NTGE[int(num_trainning // 2)] + sorted_NTGE[int(num_trainning // 2) - 1]) / 2
                    else:
                        median_NTGE = sorted_NTGE[int(num_trainning // 2)]
                else:
                    median_NTGE = "some_fails_run_more."
                np.save(best_root + 'median_NTGE', median_NTGE)
                print("median_NTGE:", median_NTGE)
                np.save(best_root + 'best_NTGE', NTGE_best_among_best)
                print("best_among_best_NTGE:", NTGE_best_among_best)
            else:
                if not os.path.exists(best_root+'best_NTGE.npy'):
                    print("dont have path:", best_root+'best_NTGE.npy')
                    print()
                    continue
                NTGE_best_among_best = np.load(best_root+'best_NTGE.npy', allow_pickle=True)
                print("best_among_best_NTGE:", NTGE_best_among_best)
                median_NTGE = np.load(best_root+'median_NTGE.npy', allow_pickle=True)
                print("median_NTGE:", median_NTGE)
