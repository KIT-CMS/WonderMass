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


def build_model(inputs, outputs_t):
    nodes = 500
    num_layer = 4
    activation = "relu"
    l2reg = None#l2(1e-2)
    #drop_p = 0.3

    # Main network
    inputs_a = Input(shape=(len(inputs),))
    net = inputs_a
    for i in range(num_layer):
        net = Dense(nodes, activation=activation, kernel_regularizer=l2reg)(net)
        net = BatchNormalization()(net)
        #net = Dropout(drop_p)(net)
    f = Dense(len(outputs_t), activation="linear", name="f")(net)

    model = Model(inputs=(inputs_a), outputs=f)
    model.summary()

    return model


def loss_mass(y_true, y_pred):
    m2 = 1.77**2

    def p3(y, offset):
        return K.square(y[:, 0 + offset]) \
             + K.square(y[:, 1 + offset]) \
             + K.square(y[:, 2 + offset])

    def p2n(y, idx):
        return K.square(y[:, idx] + y[:, idx + 3])

    def mass(y):
        p2_t1 = p3(y, 0)
        p2_t2 = p3(y, 3)
        e_t1 = K.sqrt(m2 + p2_t1)
        e_t2 = K.sqrt(m2 + p2_t2)

        e2 = K.square(e_t1 + e_t2)
        p2x = p2n(y, 0)
        p2y = p2n(y, 1)
        p2z = p2n(y, 2)

        return K.sqrt(tf.clip_by_value(
            e2 - p2x - p2y - p2z,
            0, 1e9))

    mass_true = mass(y_true)
    mass_pred = mass(y_pred)
    return 100 * K.square((mass_true - mass_pred) / mass_true)


def loss_p(y_true, y_pred):
    def p(y):
        px = y[:, 0] + y[:, 3]
        py = y[:, 1] + y[:, 4]
        #pz = y[:, 2] + y[:, 5]
        return K.sqrt(K.square(px) + K.square(py))# + K.square(pz))

    p_true = p(y_true)
    p_pred = p(y_pred)
    return 100 * K.square((p_true - p_pred) / p_true)


def loss_eta(y_true, y_pred):
    def p(y):
        px = y[:, 0] + y[:, 3]
        py = y[:, 1] + y[:, 4]
        pz = y[:, 2] + y[:, 5]
        return K.sqrt(K.square(px) + K.square(py) + K.square(pz))

    def costheta(y):
        pz = y[:, 2] + y[:, 5]
        mag = p(y)
        return tf.clip_by_value(pz / mag, 1e-9, 1e9)

    def eta(y):
        c = costheta(y)
        return -0.5 * K.log(tf.clip_by_value((1.0 - c) / (1.0 + c), 1e-5, 1e9))

    eta_true = eta(y_true)
    eta_pred = eta(y_pred)
    return 100 * K.square(eta_true - eta_pred)


def loss_phi(y_true, y_pred):
    def phi(y):
        px = y[:, 0] + y[:, 3]
        py = y[:, 1] + y[:, 4]
        return tf.atan2(py, px)

    phi_true = phi(y_true)
    phi_pred = phi(y_pred)
    return 10 * K.square(tf.clip_by_value(phi_true - phi_pred, -np.pi, np.pi))


def mse(y_true, y_pred):
    return 1e-2 * K.mean(K.square(y_pred - y_true), axis=-1)


def loss(y_true, y_pred):
    return loss_mass(y_true, y_pred) + loss_p(y_true, y_pred) + loss_eta(y_true, y_pred) + loss_phi(y_true, y_pred) + mse(y_true, y_pred)


def compile_model(model):
    model.compile(optimizer="adam",
                  loss = loss,
                  metrics = [loss_mass, loss_p, loss_eta, loss_phi, mse])


def main():
    x_train = np.load(open(save_path+"x_train_resampled.npy", "rb"))
    x_test = np.load(open(save_path+"x_test_resampled.npy", "rb"))
    y_train = np.load(open(save_path+"y_t_train_resampled.npy", "rb"))
    y_test = np.load(open(save_path+"y_t_test_resampled.npy", "rb"))

    inputs = pickle.load(open(save_path+"x.pickle", "rb"))
    outputs_t = pickle.load(open(save_path+"y_t.pickle", "rb"))
    print("Inputs: {}".format(inputs))
    print("Outputs: {}".format(outputs_t))

    model = build_model(inputs, outputs_t)
    compile_model(model)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.80, random_state=1234)

    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        batch_size=10000, epochs=10000,
                        callbacks=[EarlyStopping(patience=50),
                                   ModelCheckpoint(
                                       filepath=save_path+"model.h5", save_best_only=True, verbose=1)],
                        shuffle=True, verbose=1)

    pickle.dump({h: history.history[h] for h in history.history}, open(save_path+"history.pickle", "wb"))

    #save prediction of masses for test dataset, is used for the plotting_results.py script
    p = model.predict(x_test)
    pred = {}
    truth = {}
    for i, k in enumerate(outputs_t):
        pred[k] = p[:,i]
        truth[k] = y_test[:,i]

    tau_mass=1.77
    for d in [pred, truth]:
         for i in ["1", "2"]:
             d["t%s_gen_e"%i] = np.sqrt(tau_mass**2 + d["t%s_gen_px"%i]**2 + d["t%s_gen_py"%i]**2 + d["t%s_gen_pz"%i]**2)
         for k in ["px", "py", "pz", "e"]:
             d["h_gen_%s"%k] = d["t1_gen_%s"%k] + d["t2_gen_%s"%k]
         d["h_gen_mass"] = np.sqrt(np.abs(d["h_gen_e"]**2 - d["h_gen_px"]**2\
                                            - d["h_gen_py"]**2 - d["h_gen_pz"]**2))

    np.save(open(save_path+"mass_test_true","wb"),truth["h_gen_mass"])
    np.save(open(save_path+"mass_test_pred","wb"),pred["h_gen_mass"])


if __name__ == "__main__":
    main()
