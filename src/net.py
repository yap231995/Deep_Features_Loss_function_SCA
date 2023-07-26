from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import tensorflow as tf
##############################Model ############################################
from src.loss_fn import Soft_nearest_neighbour, Center_Loss


def mlp(extra_loss_param, length, classes=256, lr=0.0001):
    img_input = Input(shape=(length,))
    intermediate_output = []
    x = Dense(100, activation='relu')(img_input)
    intermediate_output.append(x)
    x = Dense(100, activation='relu')(x)
    intermediate_output.append(x)
    x = Dense(classes, activation='softmax')(x)
    intermediate_output.append(x)
    if extra_loss_param[0] == "center_loss":
        extra_loss_param.append(classes)
        extra_loss_param.append(intermediate_output[-2].shape[1])
    model = Custom_Model(extra_loss_param,img_input, intermediate_output)

    return model


def cnn(extra_loss_param,length, lr=0.00001, classes=256):
    img_input = Input(shape=(length, 1))
    intermediate_output = []
    x = Conv1D(2, 25, kernel_initializer='he_uniform', activation='selu', padding='same')(img_input)
    intermediate_output.append(x)
    x = AveragePooling1D(4, strides=4)(x)
    intermediate_output.append(x)
    x = Flatten(name='flatten')(x)
    intermediate_output.append(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu')(x)
    intermediate_output.append(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu')(x)
    intermediate_output.append(x)
    x = Dense(4, kernel_initializer='he_uniform', activation='selu')(x)
    intermediate_output.append(x)
    x = Dense(classes, activation='softmax')(x)
    intermediate_output.append(x)
    if extra_loss_param[0] == "center_loss":
        extra_loss_param.append(classes)
        extra_loss_param.append(intermediate_output[-2].shape[1])
    model = Custom_Model(extra_loss_param,img_input, intermediate_output)
    model.summary()
    return model





class Custom_Model(tf.keras.Model):
    def __init__(self, loss_parameters, *args, **kwargs):
        super(Custom_Model, self).__init__(*args, **kwargs)
        self.loss_type = loss_parameters[0]
        if self.loss_type == "soft_nn" or self.loss_type == "flr_soft_nn":
            self.temperature = loss_parameters[1]
            self.loss_obj = Soft_nearest_neighbour(temperature = self.temperature, distance_fn = "euclidean", name = "soft_nearest_neighbour")
        elif self.loss_type == "center_loss" or self.loss_type == "flr_center_loss":
            self.alpha = loss_parameters[1]
            self.n_classes = loss_parameters[2]
            self.n_features = loss_parameters[3]
            self.loss_obj = Center_Loss(self.n_classes, self.n_features, alpha = self.alpha, update_center = True, name = "center_loss")
        elif self.loss_type == "categorical_crossentropy" or self.loss_type == "flr":
            self.loss_obj = None
        print(self.loss_obj)



