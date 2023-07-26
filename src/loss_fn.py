import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def categorical_focal_loss(alpha, gamma=2.):
    """
    https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # print("y_pred.shape: ", y_pred.shape)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed




# Focal loss ratio,
def flr_loss(n, alpha=0.25, gamma=2.0):
    fl = categorical_focal_loss(alpha=alpha
                                    , gamma=gamma)

    def flr(y_true, y_pred):
        ce = fl(y_true, y_pred)

        ce_shuffled = 0.0

        for i in range(n):
            y_true_shuffled = tf.random.shuffle(y_true)
            ce_shuffled += fl(y_true_shuffled, y_pred)

        ce_shuffled = ce_shuffled / n

        ce_shuffled = tf.math.maximum(ce_shuffled, 1e-18)
        flr_value = ce / (ce_shuffled + 1e-40)
        return flr_value

    return flr



class FLR_LOSS(tf.keras.losses.Loss):
    def __init__(self, n, alpha= 0.25,gamma = 2.0, name="flr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n = n
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        fl = categorical_focal_loss(alpha=self.alpha
                                        , gamma=self.gamma)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        # print("ce", ce.shape)
        ce_shuffled = 0.0

        for i in range(self.n):
            y_true_shuffled = tf.random.shuffle(y_true)
            ce_shuffled += fl(y_true_shuffled, y_pred)
        # print("ce_shuffled", ce_shuffled.shape)
        ce_shuffled = ce_shuffled / self.n
        flr = ce/(ce_shuffled + 1e-40)
        # flr = tf.math.maximum(flr, 1e-18)
        #This is just to check
        # batch_size = tf.shape(y_true)[0]
        # batch_size_total = tf.cast(batch_size, dtype=y_true.dtype)
        # flr_2 = tf.reduce_sum(flr)
        # flr_2 = tf.math.divide(flr_2, batch_size_total)
        # print("flr inside:", flr_2)
        return flr


'''
Obtain the center loss by modifying the code from https://www.idiap.ch/software/bob/docs/bob/bob.learn.tensorflow/master/_modules/bob/learn/tensorflow/losses/center_loss.html.
'''

class Center_Loss(tf.keras.losses.Loss):
    def __init__(self, n_classes, n_features, alpha = 0.1, update_center = True, name = "center_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.n_features = n_features
        self.alpha = alpha
        self.update_centers = update_center
        self.centers = tf.Variable(tf.zeros([n_classes, n_features]),name="centers",
            trainable=False,)

            # in a distributed strategy, we want updates to this variable to be summed.
            #aggregation=tf.VariableAggregation.SUM)
        print("Alpha:",self.alpha)
        print("n_features:",self.n_features)

    def call(self, x_batch,labels_batch):
        # labels_batch = tf.reshape(labels_batch, (-1,))
        labels_batch = tf.math.argmax(labels_batch, axis = 1)
        x_batch = tf.reshape(x_batch,(tf.shape(x_batch)[0], -1))
        centers_batch = tf.gather(self.centers, labels_batch)
        # print("centers_batch", centers_batch)
        # the reduction of batch dimension will be done by the parent class
        center_loss = tf.keras.losses.mean_squared_error(x_batch, centers_batch)

        ### THIS IS TO CHECK ####
        # print("center loss", center_loss)
        # batch_size = tf.shape(x_batch)[0]
        # batch_size_total = tf.cast(batch_size, dtype=x_batch.dtype)
        # print("batch_size_total: ", batch_size_total)
        # center_loss_2 = tf.reduce_sum(center_loss)
        # center_loss_2 = tf.math.divide(center_loss_2, batch_size_total)
        # print("center_loss_2: ", center_loss_2)
        # print("self.centers : ", self.centers)
        return center_loss

    def update_center(self,x_batch,labels_batch):#,  update_center = True):
        # self.update_centers = update_center
        # print("Updating")
        # print("self.centers before ", self.centers)
        labels_batch = tf.math.argmax(labels_batch, axis=1)
        centers_batch = tf.gather(self.centers, labels_batch)
        diff = (centers_batch - x_batch)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels_batch)
        # print("unique_label", unique_label)
        # print("unique_idx", unique_idx)
        # print("unique_count", unique_count)
        appear_times = tf.gather(unique_count, unique_idx)
        # print("appear_times 1", appear_times)
        appear_times = tf.reshape(appear_times, [-1, 1])
        # print("appear_times 2", appear_times)
        batch_size = tf.shape(x_batch)[0]
        # find all same class neighbour
        pos_mask, _ = self.build_masks(labels_batch, labels_batch, batch_size, remove_diagonal=False)
        # print("diff: ", diff)
        # print("pos_mask: ", pos_mask)
        pos_mask = tf.cast(pos_mask, dtype=x_batch.dtype)
        sacn_diff = tf.tensordot(pos_mask, diff, axes=1)
        # print("sacn_diff", sacn_diff)

        sacn_diff = tf.math.divide(sacn_diff, tf.cast((1 + appear_times), tf.float32))
        sacn_diff = self.alpha * sacn_diff
        updates = tf.scatter_nd(indices=labels_batch[:, None], updates=sacn_diff,
                                shape=self.centers.shape)  # This will cause: update = [1,2,3] -> [0,1,0,0,2,3], with indices = [1,4,5] and shape = (6,)
        # print("updates", updates)
        # using assign_sub will make sure updates are added during distributed
        # training
        self.centers.assign_sub(updates)  # this will do the following: self.center = self.center - updates
        # print("self.centers after ", self.centers)

    def build_masks(self, query_labels, key_labels, batch_size, remove_diagonal=True):
        """Build masks that allows to select only the positive or negatives
        embeddings.
        Args:
            query_labels: 1D int `Tensor` that contains the query class ids.
            key_labels: 1D int `Tensor` that contains the key class ids.
            batch_size: size of the batch.
            remove_diagonal: Bool. If True, will set diagonal to False in positive pair mask
        Returns:
            Tuple of Tensors containing the positive_mask and negative_mask
        """
        if tf.rank(query_labels) == 1:
            # if len(query_labels.shape) == 1:
            query_labels = tf.reshape(query_labels, (-1, 1))

        if tf.rank(key_labels) == 1:
            # if len(query_labels.shape) == 1:
            key_labels = tf.reshape(key_labels, (-1, 1))
        # print("query_labels inside build_mask 2", query_labels)
        # print("key_labels inside build_mask 2", key_labels)
        # same class mask
        positive_mask = tf.math.equal(query_labels, tf.transpose(key_labels))

        # not the same class
        negative_mask = tf.math.logical_not(positive_mask)

        if remove_diagonal:
            # It is optional to remove diagonal from positive mask.
            # Diagonal is often removed if queries and keys are identical.
            positive_mask = tf.linalg.set_diag(positive_mask, tf.zeros(batch_size, dtype=tf.bool))
        return positive_mask, negative_mask


'''
Obtain the soft nearest neighbour loss by modifying the code from https://github.com/tensorflow/similarity.
'''
class Soft_nearest_neighbour(tf.keras.losses.Loss):
    def __init__(self, temperature = 2, distance_fn = "euclidean", name = "soft_nearest_neighbour", **kwargs):
        super().__init__(name=name, **kwargs)
        self.t_placeholder = tf.Variable(1., dtype=tf.float32, trainable=False, name="temp")
        # self.initial_temperature = 1
        self.temperature = temperature #Controls relative importance given to the pair of points.
        self.distance_fn = distance_fn #distance function to compute the pairwise
        print("Temperature:", self.temperature)
    def call(self, x_batch,labels_batch):
        labels_batch = tf.math.argmax(labels_batch, axis=1)
        x_batch = tf.reshape(x_batch,(tf.shape(x_batch)[0], -1))
        # print(x_batch)
        if self.distance_fn == 'euclidean': #This is square euclidean
            x_squared_norm = tf.math.square(x_batch)
            x_squared_norm = tf.math.reduce_sum(x_squared_norm, axis = 1, keepdims = True)
            distances = 2.0 * tf.linalg.matmul(x_batch, x_batch, transpose_b=True)
            distances = x_squared_norm - distances + tf.transpose(x_squared_norm) #this is the expanded form of sum (p-q)^2
            # Avoid NaN and inf gradients when back propagating through the sqrt.
            # values smaller than 1e-18 produce inf for the gradient, and 0.0
            # produces NaN. All values smaller than 1e-13 should produce a gradient
            # of 1.0.
            distances = tf.math.maximum(distances, 1e-18)
        # print(distances)
        batch_size = tf.shape(x_batch)[0]
        eps = tf.cast(1e-9, dtype=x_batch.dtype)
        distances = distances/self.temperature
        # print("distances: ",distances)
        negexpd = tf.math.exp(-distances)
        negexpd = tf.math.maximum(negexpd, 1e-18) #This line makes the soft nn loss stable. If not will cause NaN as stated above.
        # print("negexpd: ",negexpd)
        # Mask out diagonal entries
        diag = tf.linalg.diag(tf.ones(batch_size, dtype=tf.bool))
        diag_mask = tf.cast(tf.logical_not(diag), dtype=x_batch.dtype)
        # print(diag_mask)
        negexpd = tf.math.multiply(negexpd, diag_mask)
        # creating mask to sample same class neighboorhood (note: remove the diagonal.)
        pos_mask, _ = self.build_masks(
            labels_batch,
            labels_batch,
            batch_size=batch_size,
            remove_diagonal=True,
        )
        # print(pos_mask)
        pos_mask = tf.cast(pos_mask, dtype=x_batch.dtype)
        # all class neighborhood
        alcn = tf.reduce_sum(negexpd, axis=1)

        # print("alcn: ", alcn)
        # same class neighborhood
        sacn = tf.reduce_sum(tf.math.multiply(negexpd, pos_mask), axis=1)
        # print("sacn: ", sacn)
        softnn_loss = tf.math.divide(sacn, alcn)
        # print("sacn/alcn: ", softnn_loss)

        softnn_loss = tf.math.log(eps + softnn_loss)
        # print("log(sacn/alcn): ", softnn_loss)
        softnn_loss = tf.math.multiply(softnn_loss, -1)
        ### THIS IS TO CHECK ####
        # softnn_loss_2 = tf.reduce_sum(softnn_loss)
        # # print("sum log(sacn/alcn): ", softnn_loss_2)
        # batch_size_total = tf.cast(batch_size, dtype=x_batch.dtype)
        # # print("batch_size_total: ", batch_size_total)
        # softnn_loss_2 = tf.math.divide(softnn_loss_2, batch_size_total)
        # print("1/b sum log(sacn/alcn) final: ", softnn_loss_2)

        return softnn_loss



    def build_masks(self, query_labels, key_labels, batch_size, remove_diagonal=True):
        """Build masks that allows to select only the positive or negatives
        embeddings.
        Args:
            query_labels: 1D int `Tensor` that contains the query class ids.
            key_labels: 1D int `Tensor` that contains the key class ids.
            batch_size: size of the batch.
            remove_diagonal: Bool. If True, will set diagonal to False in positive pair mask
        Returns:
            Tuple of Tensors containing the positive_mask and negative_mask
        """
        if tf.rank(query_labels) == 1:
            # if len(query_labels.shape) == 1:
            query_labels = tf.reshape(query_labels, (-1, 1))

        if tf.rank(key_labels) == 1:
            # if len(query_labels.shape) == 1:
            key_labels = tf.reshape(key_labels, (-1, 1))
        # print("query_labels inside build_mask 2", query_labels)
        # print("key_labels inside build_mask 2", key_labels)
        # same class mask
        positive_mask = tf.math.equal(query_labels, tf.transpose(key_labels))

        # not the same class
        negative_mask = tf.math.logical_not(positive_mask)

        if remove_diagonal:
            # It is optional to remove diagonal from positive mask.
            # Diagonal is often removed if queries and keys are identical.
            positive_mask = tf.linalg.set_diag(positive_mask, tf.zeros(batch_size, dtype=tf.bool))
        return positive_mask, negative_mask
