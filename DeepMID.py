import tensorflow
from tensorflow.keras import Input, layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import pickle, os
from sklearn.metrics import confusion_matrix
import tensorflow as tf

import numpy as np
from tensorflow.python.framework import ops
from sklearn import preprocessing
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

import random


def create_input_layers(xshapes):
    inputs = []
    for xshape in xshapes:
        input_shape_x = (xshape[1], 1)
        input_x = Input(shape = input_shape_x)
        inputs.append(input_x)
    return inputs


def create_convolution_layers(inputs, num_layers = 0):
    convs = []
    for input_x in inputs:
        conv      = layers.Conv1D(32,  5,  kernel_initializer = 'he_normal', input_shape=input_x.get_shape())(input_x)
        conv      = layers.Activation('relu')(conv)
        conv      = layers.MaxPooling1D(strides=2, padding='valid')(conv)
        # print('1 layer =',conv.shape)
        for i in range(num_layers):
            conv      = layers.Conv1D(32,  5,  kernel_initializer = 'he_normal')(conv)
            conv      = layers.Activation('relu')(conv)
            conv      = layers.MaxPooling1D(strides=2, padding='valid')(conv)
        convs.append(conv)
    return convs


class SpatialPyramidPooling(Layer):

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[3]


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)


        num_rows = input_shape[1]
        num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []


        for pool_num, num_pool_regions in enumerate(self.pool_list):

            for ix in range(num_pool_regions):
                for iy in range(num_pool_regions):
                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]
                    y1 = iy * row_length[pool_num]
                    y2 = iy * row_length[pool_num] + row_length[pool_num]

                    x1 = K.cast(K.round(x1), 'int32')
                    x2 = K.cast(K.round(x2), 'int32')
                    y1 = K.cast(K.round(y1), 'int32')
                    y2 = K.cast(K.round(y2), 'int32')

                    new_shape = [input_shape[0], y2 - y1,
                                 x2 - x1, input_shape[3]]
                    x_crop = x[:, y1:y2, x1:x2, :]
                    xm = K.reshape(x_crop, new_shape)
                    pooled_val = K.max(xm, axis=(1, 2))
                    outputs.append(pooled_val)

        if self.dim_ordering == 'channels_first':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'channels_last':
            outputs = K.concatenate(outputs,axis = 0)
            outputs = K.reshape(outputs,(self.num_outputs_per_channel,input_shape[0], self.nb_channels))
            outputs = K.permute_dimensions(outputs,(1,0,2))
            outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))
        return outputs

def MICNN(xshapes, num_conv_layers, lr):
    inputs = create_input_layers(xshapes)
    convs = create_convolution_layers(inputs, num_layers = num_conv_layers)
    #有SPP层
    if len(convs) >= 2:
        conv_merge = layers.concatenate(convs,2)
        print('conv_merge layer =',conv_merge.shape)
        conv_merge = tf.expand_dims(conv_merge, -1)

        conv1 = tf.keras.layers.Conv2D(128, kernel_size=(5,5), strides=(2,2),padding = 'same')(conv_merge)
        conv1 = tf.keras.layers.Activation('relu')(conv1)
        print('conv1 layer =',conv1.shape)
    else:
        conv_merge = convs[0]
    #flat      = layers.Flatten()(conv1)
    spp = SpatialPyramidPooling([1, 2, 3, 4])(conv1)
    dense     = layers.Dense(100,  activation='relu')(spp)
    dense     = layers.Dropout(0.2)(dense)
    output    = layers.Dense(1, activation='sigmoid')(dense)
    model     = models.Model(inputs= inputs, outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_MICNN(model, Xs, y, batch, epochs, Xs_valid, y_valid, callbacks=None):
    Xs3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    Xs3d_valid = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs_valid]
    model.fit(Xs3d, y, batch_size=batch, epochs=epochs, validation_data=(Xs3d_valid, y_valid), callbacks=callbacks)


def plot_loss_accuracy(model):
    history = model.history.history
    linewidth = 1
    fig = plt.figure(figsize=(6, 4.5))
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xlabel('Epoch')
    plt.plot(history['accuracy'], color='g', linewidth=linewidth, label='training_accuracy')
    plt.plot(history['val_accuracy'], color='b', linewidth=linewidth, label='validation_accuracy')
    plt.legend( loc='upper right',bbox_to_anchor=(0.98,0.9), fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', fontsize=12)
    plt.plot(history['loss'], color='r', linewidth=linewidth, label='training_loss')
    plt.plot(history['val_loss'], color='c', linewidth=linewidth, label='validation_loss')
    plt.legend( loc='lower right',bbox_to_anchor=(0.98,0.1), fontsize=12)
    plt.ylabel('Loss', fontsize=12)


def save_MICNN(model, model_name):
    model_path = f'{model_name}.h5'
    history_path = f'{model_name}.pkl'
    model.save(model_path)
    pickle.dump(model.history.history, open(history_path, "wb" ))


def load_MICNN(model_name):
    model_path = f'{model_name}.h5'
    history_path = f'{model_name}.pkl'
    model = models.load_model(model_path,custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})
    history = pickle.load(open(history_path, "rb" ))
    model.history = callbacks.History()
    model.history.history = history
    return model


def check_MICNN(model_name):
    model_path = f'{model_name}.h5'
    history_path = f'{model_name}.pkl'
    return os.path.isfile(model_path) and os.path.isfile(history_path)


def predict_MICNN(model, Xs):
    Xs3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    return model.predict(Xs3d)


def evaluate_MICNN(model, Xs, y):
    Xs3d = [X.reshape((X.shape[0], X.shape[1], 1)) for X in Xs]
    return model.evaluate(Xs3d, y)


if __name__ == "__main__":
    model_save_path = '/model/model_1/'
    model_name = 'test_nmr'
    # bTrain = True # train model
    bTrain = False # predict result
    if bTrain:
        pickle_file = open(f'aug/data_augment_train.pkl', 'rb')
        aug = pickle.load(pickle_file)

        pickle_file_valid = open(f'aug/data_augment_valid.pkl', 'rb')
        aug_valid = pickle.load(pickle_file_valid)

        model = MICNN([aug['R'].shape, aug['S'].shape], 8)
        # MICNN(xshapes, num_conv_layers-1)
        train_MICNN(model, [aug['R'], aug['S']], aug['y'], 64, 100, [aug_valid['R'], aug_valid['S']], aug_valid['y'])
        # train_MICNN(model, Xs, y, batch, epochs, Xs_valid, y_valid):
        save_MICNN(model, model_save_path + model_name)

        model = load_MICNN(model_save_path + model_name)
        plot_loss_accuracy(model)

        stds = read_bruker_hs_base('data/standards_13', False, True, False)
        spectra = read_bruker_hs_base('data/known', False, True, False)

        #test set
        pickle_file_test = open('aug/data_augment_test.pkl', 'rb')
        aug_test = pickle.load(pickle_file_test)
        ev = evaluate_MICNN(model, [aug_test['R'], aug_test['S']], aug_test['y'])
        yp_test = predict_MICNN(model, [aug_test['R'], aug_test['S']])
        yp_test_list = [1 if yp_test[i, 0] >= 0.5 else 0 for i in range(yp_test.shape[0])]
        yp_test = np.array(yp_test_list).reshape(yp_test.shape)
        cnf_matrix = confusion_matrix(aug_test['y'], yp_test)
        np.savetxt('PGzl1.13_1.27_8_10-5_64_200_23flavors_cnf_matrix.csv', cnf_matrix, delimiter=',')
        # known formulated flavors
        for i in range(16):
            query = spectra[i]

            p = query['ppm'].shape[0]
            n = len(stds)
            R = np.zeros((n, p), dtype=np.float32)
            Q = np.zeros((n, p), dtype=np.float32)
            for i in range(n):
                R[i,] = stds[i]['fid']
                Q[i,] = query['fid']
            yp = predict_MICNN(model, [R, Q])

            stds_df = pd.read_csv('data/standard_13.csv', encoding='gb2312')
            result_df = pd.DataFrame(columns=['Name', 'Probability'])
            for t in range(n):
                result_df.loc[len(result_df)] = [stds[t]['name'], yp[t][0]]

            result = pd.merge(stds_df, result_df, on=['Name'])
            result1 = result.sort_values(by=['Probability'], ascending=False)
            outputpath = "{} {} {}".format(
                "result/model_1_known_formulated_flavors_result_",
                query['name'], ".csv")
            result1.to_csv(outputpath, sep=',', encoding='utf_8_sig', index=True, header=True)

    else:
        model = load_MICNN(model_save_path + model_name)
        plot_loss_accuracy(model)

        stds = read_bruker_hs_base('data/standards_13', False, True, False)
        spectra = read_bruker_hs_base('data/known', False, True, False)

        #test set
        pickle_file_test = open('aug/data_augment_test.pkl', 'rb')
        aug_test = pickle.load(pickle_file_test)
        ev = evaluate_MICNN(model, [aug_test['R'], aug_test['S']], aug_test['y'])
        yp_test = predict_MICNN(model, [aug_test['R'], aug_test['S']])
        yp_test_list = [1 if yp_test[i, 0] >= 0.5 else 0 for i in range(yp_test.shape[0])]
        yp_test = np.array(yp_test_list).reshape(yp_test.shape)
        cnf_matrix = confusion_matrix(aug_test['y'], yp_test)
        np.savetxt('PGzl1.13_1.27_8_10-5_64_200_23flavors_cnf_matrix.csv', cnf_matrix, delimiter=',')
        # known formulated flavors
        for i in range(16):
            query = spectra[i]

            p = query['ppm'].shape[0]
            n = len(stds)
            R = np.zeros((n, p), dtype=np.float32)
            Q = np.zeros((n, p), dtype=np.float32)
            for i in range(n):
                R[i,] = stds[i]['fid']
                Q[i,] = query['fid']
            yp = predict_MICNN(model, [R, Q])

            stds_df = pd.read_csv('data/standard_13.csv', encoding='gb2312')
            result_df = pd.DataFrame(columns=['Name', 'Probability'])
            for t in range(n):
                result_df.loc[len(result_df)] = [stds[t]['name'], yp[t][0]]

            result = pd.merge(stds_df, result_df, on=['Name'])
            result1 = result.sort_values(by=['Probability'], ascending=False)
            outputpath = "{} {} {}".format(
                "result/model_1_known_formulated_flavors_result_",
                query['name'], ".csv")
            result1.to_csv(outputpath, sep=',', encoding='utf_8_sig', index=True, header=True)