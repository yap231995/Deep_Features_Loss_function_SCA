import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from src.loss_fn import FLR_LOSS



def evaluate(inputs, labels, model, lamb, loss_fn):
    '''
    Takes a single step of training. (one epoch, input is one batch)

    Args:
        inputs: A tensor. A batch of input to the model.
        labels: A tensor. The labels to use when training.
        model: A tf.keras.Model object, or subclass thereof.
        lamb: regularization strength
        loss_fn: loss function for the
        optimizer
    Returns:
        The predictions of the model on the inputs. Useful if you need to update metrics after training.
    '''
    # model.loss_obj.temperature_optimized(inputs, labels,
    #                                      model.loss_obj.temperature)  # Optimize the temperature

    predictions_with_intermediate_layer = model(inputs, training=False)
    predictions = predictions_with_intermediate_layer[-1]
    intermediate_layers_output = predictions_with_intermediate_layer[:-1]
    loss = 0
    if model.loss_type == "soft_nn" or model.loss_type == "flr_soft_nn":
        for one_layer_output in intermediate_layers_output:
            soft_nn_loss = model.loss_obj(one_layer_output, labels)
            # print("val soft_nn_loss outside", soft_nn_loss)
            loss += soft_nn_loss
    elif model.loss_type == "center_loss" or model.loss_type == "flr_center_loss":
        loss += model.loss_obj(intermediate_layers_output[-1], labels)
        # print(labels.shape)
        # print(predictions.shape)
    pred_loss = loss_fn(labels, predictions)
    total_loss = pred_loss
    total_loss += lamb * loss
    if len(model.losses) > 0:
        regularization_loss = tf.math.add_n(model.losses)
        total_loss = total_loss + regularization_loss
    return predictions, total_loss


@tf.function
def train_step(inputs, labels, model, lamb, loss_fn, optimizer):
    '''
    Takes a single step of training. (one epoch, input is one batch)

    Args:
        inputs: A tensor. A batch of input to the model.
        labels: A tensor. The labels to use when training.
        model: A tf.keras.Model object, or subclass thereof.
        lamb: regularization strength
        loss_fn: loss function for the
        optimizer
    Returns:
        The predictions of the model on the inputs. Useful if you need to update metrics after training.
    '''
    # model.loss_obj.temperature_optimized(inputs, labels,
    #                                      model.loss_obj.temperature)  # Optimize the temperature
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions_with_intermediate_layer = model(inputs, training=True)
        predictions = predictions_with_intermediate_layer[-1]
        intermediate_layers_output = predictions_with_intermediate_layer[:-1]
        # loss = tf.constant(0.)
        loss = 0
        if model.loss_type == "soft_nn" or model.loss_type == "flr_soft_nn":
            # print("soft_nn being activated")
            for one_layer_output in intermediate_layers_output:
                soft_nn_loss = model.loss_obj(one_layer_output,labels)
                # print("train soft_nn_loss outside", soft_nn_loss)
                loss += soft_nn_loss
        elif model.loss_type == "center_loss" or model.loss_type == "flr_center_loss":
            # print("center_loss being activated")
            center_loss = model.loss_obj(intermediate_layers_output[-1], labels)
            # print("center_loss outside", center_loss)
            loss += center_loss
        # print(labels.shape)
        # print(predictions.shape)
        pred_loss = loss_fn(labels, predictions)
        # print("cer",pred_loss)
        total_loss = pred_loss
        total_loss += lamb * loss
        if len(model.losses) > 0:
            regularization_loss = tf.math.add_n(model.losses)
            total_loss = total_loss + regularization_loss


    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if model.loss_type == "center_loss" or model.loss_type == "flr_center_loss":
        model.loss_obj.update_center(intermediate_layers_output[-1], labels)
    return predictions, total_loss




def train(x_train,y_train, x_val, y_val,model, num_epochs,batch_size,lamb,optimizer, base_loss_fn = 'categorical_crossentropy'):
    print("lambda:", lamb)
    if base_loss_fn == 'categorical_crossentropy':
        print("BASE_LOSS_FN:", base_loss_fn)
        print("loss_type:", model.loss_type)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
    elif base_loss_fn == 'flr':
        print("BASE_LOSS_FN:", base_loss_fn)
        print("loss_type:", model.loss_type)
        loss_fn = FLR_LOSS(10)
    train_acc_fn = tf.keras.metrics.CategoricalAccuracy()
    val_acc_fn = tf.keras.metrics.CategoricalAccuracy()

    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []
    for epoch in range(num_epochs):
        start_time = time.time()
        indices = np.random.permutation(len(x_train)).astype(int)
        epoch_train_loss = 0
        epoch_val_loss = 0
        n_batch = 0
        n_batch_val = 0
        for i in range(0, len(x_train), batch_size):
            x_batch_train = tf.convert_to_tensor(x_train[indices[i:min(i + batch_size, len(x_train))], :, :])
            y_batch_train = tf.convert_to_tensor(y_train[indices[i:min(i + batch_size, len(y_train))]])
            predictions,total_loss = train_step(x_batch_train, y_batch_train, model, lamb, loss_fn,optimizer)
            train_acc_fn(y_batch_train, predictions)
            # print("total_loss_val:", total_loss)
            # print("epoch_val_loss:", epoch_train_loss)
            epoch_train_loss += total_loss
            n_batch += 1
        epoch_train_loss = epoch_train_loss/n_batch
        train_acc = train_acc_fn.result().numpy()

        for i in range(0, len(x_val), batch_size):
            x_batch_val = x_val[i:min(i + batch_size, len(x_val))]
            y_batch_val = y_val[i:min(i + batch_size, len(y_val))]
            # test_logits = model(x_batch_val)
            predictions_val, total_loss_val = evaluate(x_batch_val, y_batch_val, model, lamb, loss_fn)
            # print("total_loss_val:", total_loss_val)
            # print("epoch_val_loss:", epoch_val_loss)
            epoch_val_loss += total_loss_val
            n_batch_val += 1
            val_acc_fn(y_batch_val, predictions_val)
        epoch_val_loss = epoch_val_loss / n_batch_val
        val_acc = val_acc_fn.result().numpy()

        print('Epoch {} - train_acc: {:.4f} , val_acc: {:.4f}, train_loss: {:.4f} , val_loss: {:.4f} ({:.1f} seconds / epoch)'.format(epoch + 1,
                                                                                  train_acc,
                                                                                   val_acc,epoch_train_loss,epoch_val_loss, time.time() - start_time))
        all_train_loss.append(epoch_train_loss)
        all_val_loss.append(epoch_val_loss)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)

        train_acc_fn.reset_states()
        val_acc_fn.reset_states()
    history = {"train_loss": all_train_loss,"val_loss": all_val_loss, "train_accuracy": all_train_acc, "val_accuracy": all_val_acc}
    return history






def evaluate_embbeding(inputs, labels, model, loss_fn,loss_type):
    '''
    Takes a single step of training. (one epoch, input is one batch)

    Args:
        inputs: A tensor. A batch of input to the model.
        labels: A tensor. The labels to use when training.
        model: A tf.keras.Model object, or subclass thereof.
        lamb: regularization strength
        loss_fn: loss function for the
        optimizer
    Returns:
        The predictions of the model on the inputs. Useful if you need to update metrics after training.
    '''

    predictions = model(inputs, training=False)
    total_loss = 0
    if loss_type == "soft_nn":
        total_loss += loss_fn(predictions, labels)
    elif loss_type == "center_loss":
        # loss_fn.update_center(update_center=False)
        total_loss += loss_fn(predictions[0], labels)
        predictions = predictions[-1]
    return predictions, total_loss


@tf.function
def train_step_embedding(inputs, labels, model, loss_fn, optimizer,loss_type):
    '''
    Takes a single step of training. (one epoch, input is one batch)

    Args:
        inputs: A tensor. A batch of input to the model.
        labels: A tensor. The labels to use when training.
        model: A tf.keras.Model object, or subclass thereof.
        lamb: regularization strength
        loss_fn: loss function for the
        optimizer
    Returns:
        The predictions of the model on the inputs. Useful if you need to update metrics after training.
    '''
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=True)
        # loss = tf.constant(0.)
        total_loss = 0
        if loss_type == "soft_nn":
                soft_nn_loss =  loss_fn(predictions,labels)
                # print("soft_nn_loss outside:", soft_nn_loss)
                total_loss +=soft_nn_loss
        elif loss_type == "center_loss":
            # print(predictions)
            center_loss = loss_fn(predictions,labels)
            # loss_fn_softmax = tf.keras.losses.CategoricalCrossentropy()
            # print("center_loss outside:", center_loss)
            total_loss += center_loss
            # total_loss += loss_fn_softmax(labels, predictions[-1])

    gradients = tape.gradient(total_loss, model.trainable_variables)
    # print("gradients", gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if loss_type == "center_loss":
        loss_fn.update_center(predictions, labels)
        # predictions = predictions[-1]
    return predictions, total_loss



def train_embedding_with_validation(x_train,y_train, x_val, y_val,model, num_epochs,batch_size,optimizer,loss_type,loss_fn):
    # loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_acc_fn = tf.keras.metrics.CategoricalAccuracy()
    val_acc_fn = tf.keras.metrics.CategoricalAccuracy()

    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []
    for epoch in range(num_epochs):
        start_time = time.time()
        indices = np.random.permutation(len(x_train)).astype(int)
        epoch_train_loss = 0
        epoch_val_loss = 0
        n_batch = 0
        n_batch_val = 0
        for i in range(0, len(x_train), batch_size):
            x_batch_train = tf.convert_to_tensor(x_train[indices[i:min(i + batch_size, len(x_train))], :, :])
            y_batch_train = tf.convert_to_tensor(y_train[indices[i:min(i + batch_size, len(y_train))]])
            predictions, total_loss = train_step_embedding(x_batch_train, y_batch_train, model, loss_fn, optimizer,
                                                           loss_type)
            train_acc_fn(y_batch_train, predictions)
            epoch_train_loss += total_loss
            n_batch += 1
        epoch_train_loss = epoch_train_loss / n_batch
        train_acc = train_acc_fn.result().numpy()

        for i in range(0, len(x_val), batch_size):
            x_batch_val = x_val[i:min(i + batch_size, len(x_val))]
            y_batch_val = y_val[i:min(i + batch_size, len(y_val))]
            predictions_val, total_loss_val = evaluate_embbeding(x_batch_val, y_batch_val, model, loss_fn,loss_type)
            epoch_val_loss += total_loss_val
            n_batch_val += 1
            val_acc_fn(y_batch_val, predictions_val)
        epoch_val_loss = epoch_val_loss / n_batch_val
        val_acc = val_acc_fn.result().numpy()

        print('Epoch {} - train_acc: {:.4f} , val_acc: {:.4f}, train_loss: {:.4f} , val_loss: {:.4f} ({:.1f} seconds / epoch)'.format(epoch + 1,
                                                                                  train_acc,
                                                                                   val_acc,epoch_train_loss,epoch_val_loss, time.time() - start_time))

        all_train_loss.append(epoch_train_loss)
        all_val_loss.append(epoch_val_loss)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)

        train_acc_fn.reset_states()
        val_acc_fn.reset_states()

    history = {"train_loss": all_train_loss, "val_loss": all_val_loss, "train_accuracy": all_train_acc,
               "val_accuracy": all_val_acc}

    return history


def train_embedding(x_train,y_train,model, num_epochs,batch_size,optimizer,loss_type,loss_fn):
    # loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_acc_fn = tf.keras.metrics.CategoricalAccuracy()
    val_acc_fn = tf.keras.metrics.CategoricalAccuracy()

    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []
    for epoch in range(num_epochs):
        start_time = time.time()
        indices = np.random.permutation(len(x_train)).astype(int)
        epoch_train_loss = 0
        epoch_val_loss = 0
        n_batch = 0
        n_batch_val = 0
        val_acc = 0
        for i in range(0, len(x_train), batch_size):
            x_batch_train = tf.convert_to_tensor(x_train[indices[i:min(i + batch_size, len(x_train))], :, :])
            y_batch_train = tf.convert_to_tensor(y_train[indices[i:min(i + batch_size, len(y_train))]])
            predictions ,total_loss = train_step_embedding(x_batch_train, y_batch_train, model, loss_fn,optimizer,loss_type)
            # print("prediction:", predictions.shape)
            # print("y_batch_train:", y_batch_train.shape)
            # train_acc_fn(y_batch_train, predictions)
            epoch_train_loss += total_loss
            n_batch += 1
        epoch_train_loss = epoch_train_loss/n_batch
        train_acc = train_acc_fn.result().numpy()

        # for i in range(0, len(x_val), batch_size):
        #     x_batch_val = x_val[i:min(i + batch_size, len(x_val))]
        #     y_batch_val = y_val[i:min(i + batch_size, len(y_val))]
        #     predictions_val, total_loss_val = evaluate_embbeding(x_batch_val, y_batch_val, model, loss_fn,loss_type)
        #     epoch_val_loss += total_loss_val
        #     n_batch_val += 1
        #     val_acc_fn(y_batch_val, predictions_val)
        # epoch_val_loss = epoch_val_loss / n_batch_val
        # val_acc = val_acc_fn.result().numpy()

        # print('Epoch {} - train_acc: {:.4f} , val_acc: {:.4f}, train_loss: {:.4f} , val_loss: {:.4f} ({:.1f} seconds / epoch)'.format(epoch + 1,
        #                                                                           train_acc,
        #                                                                            val_acc,epoch_train_loss,epoch_val_loss, time.time() - start_time))
        print('Epoch {} - train_acc: {:.4f} , train_loss: {:.4f}  ({:.1f} seconds / epoch)'.format(epoch + 1,
                                                                                          train_acc,epoch_train_loss, time.time() - start_time))
        all_train_loss.append(epoch_train_loss)
        all_val_loss.append(epoch_val_loss)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)

        train_acc_fn.reset_states()
        val_acc_fn.reset_states()


    history = {"train_loss": all_train_loss,"val_loss": all_val_loss, "train_accuracy": all_train_acc, "val_accuracy": all_val_acc}

    return history




























