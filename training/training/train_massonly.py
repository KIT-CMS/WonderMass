import numpy as np
np.random.seed(1234)
import pickle

from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate, Lambda, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
import keras.backend as K
import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.tensorflow_backend.set_session(sess)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)

save_path="tempsave/"


def build_model(inputs):
    nodes = 500
    num_layer = 4
    activation = "elu"
    l2reg = None#l2(1e-2)
    drop_p = 0.3
    # Main network
    inputs_a = Input(shape=(len(inputs),))
    net = inputs_a
    for i in range(num_layer):
        net = Dense(nodes, activation=activation, kernel_regularizer=l2reg)(net)
        net = BatchNormalization()(net)
        net = Dropout(drop_p)(net)
    f = Dense(1, activation="linear", name="f")(net)

    model = Model(inputs=(inputs_a), outputs=f)
    model.summary()

    return model


def loss_mass(y_true, y_pred):
    return K.square((y_true - y_pred) / y_true)


def compile_model(model):
    model.compile(optimizer="adam",
                  loss = loss_mass,
                  metrics = [loss_mass])


def main():
    x_train = np.load(open(save_path+"x_train_resampled.npy", "rb"))
    x_val = np.load(open(save_path+"x_test_resampled.npy", "rb"))
    mass_train=np.load(open(save_path+"mass_train_npy","rb"))
    mass_val=np.load(open(save_path+"mass_test_npy","rb"))

    inputs = pickle.load(open(save_path+"x.pickle", "rb"))
    print("Inputs: {}".format(inputs))


    model = build_model(inputs)
    compile_model(model)

    #x_train, x_val, y_train, y_val = \
    #        train_test_split(x, y_t, train_size=0.80, random_state=1234)

    history = model.fit(x_train, mass_train,
                        validation_data=(x_val, mass_val),
                        batch_size=10000, epochs=10000,
                        callbacks=[EarlyStopping(patience=100),
                                   ModelCheckpoint(
                                       filepath=save_path+"model.h5", save_best_only=True, verbose=1)],
                        shuffle=True, verbose=1)

    pickle.dump({h: history.history[h] for h in history.history}, open(save_path+"history.pickle", "wb"))

    #save mass prediction for test dataset
    predict=np.array(model.predict(x_val))
    predict=np.squeeze(predict)
    np.save(open(save_path+"mass_test_pred","wb"),predict)

if __name__ == "__main__":
    main()
