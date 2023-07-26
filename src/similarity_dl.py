from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *

from src.net import Custom_Model


def Visualize_net_cnn(length, embedding_size):
    img_input = Input(shape=(length,1))
    x = Conv1D(64, 15, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    x = AveragePooling1D(15, strides=15)(x)
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same')(x)
    x = AveragePooling1D(2, strides=2)(x)
    x = Flatten(name='flatten')(x)
    # x = Dense(100, activation='relu')(x)
    # x = Dense(100, activation='relu')(x)
    x = Dense(embedding_size)(x)
    model = Model(img_input, x)

    # optimizer = RMSprop(lr=lr)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def Visualize_net_mlp(length, embedding_size, loss_type, classes):
    img_input = Input(shape=(length,))
    intermediate_output = []
    x = Dense(100, activation='relu')(img_input)
    x = Dense(100, activation='relu')(x)
    x = Dense(embedding_size)(x)
    intermediate_output.append(x)
    softmax_output = Dense(classes, activation = 'softmax')(x)
    intermediate_output.append(softmax_output)
    if loss_type == "soft_nn":
        model = Model(img_input, x)
    elif loss_type == "center_loss" or loss_type == "categorical_crossentropy" or loss_type =="flr" or loss_type == "flr_center_loss":
        extra_loss_param = [loss_type,0.01]
        extra_loss_param.append(classes)
        extra_loss_param.append(intermediate_output[0].shape[1])
        model = Custom_Model(extra_loss_param, img_input, intermediate_output)

    # optimizer = RMSprop(lr=lr)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model