import numpy as np
np.random.seed(1234)
import pickle
import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
import os
from shutil import copyfile

from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import plot_model

import keras.backend as K
import tensorflow as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.tensorflow_backend.set_session(sess)

#save_path="tempsave/"
save_path="saved_models/saves_false_train_old/"

def main():
    # Load data
    x_train = np.load(open(save_path+"x_train_resampled.npy", "rb"))
    x_val = np.load(open(save_path+"x_test_resampled.npy", "rb"))
    y_train = np.load(open(save_path+"y_t_train_resampled.npy", "rb"))
    y_val = np.load(open(save_path+"y_t_test_resampled.npy", "rb"))

    inputs = pickle.load(open(save_path+"x.pickle", "rb"))
    outputs_t = pickle.load(open(save_path+"y_t.pickle", "rb"))

    # Load model
    def dummy(y_true, y_pred):
        return y_pred[:,0]
    copyfile(save_path+"model.h5", save_path+"model_copy.h5")
    model = load_model(save_path+"model_copy.h5",
            custom_objects={
                "mass_t_loss": dummy,
                "mass_h_loss": dummy,
                "loss": dummy,
                "loss_p3": dummy,
                "loss_p3_1": dummy,
                "loss_p3_2": dummy,
                "loss_p3_h": dummy,
                "loss_p_1": dummy,
                "loss_p_2": dummy,
                "loss_p_h": dummy,
                "loss_p3_h": dummy,
                "loss_f": dummy,
                "loss_f_1": dummy,
                "loss_f_2": dummy,
                "loss_mass": dummy,
                "loss_pt": dummy,
                "loss_p": dummy,
                "loss_eta": dummy,
                "loss_phi": dummy,
                           })

    # Plot model
    #plot_model(model, to_file="model.png")

    # Split data
    #x_train, x_val, y_train, y_val = \
    #        train_test_split(x, y_t, train_size=0.80, random_state=1234)

    # Make prediction
    p = model.predict(x_val)
    pred = {}
    truth = {}
    for i, k in enumerate(outputs_t):
        pred[k] = p[:,i]
        truth[k] = y_val[:,i]

    # Add energy
    tau_mass = 1.77
    for d in [pred, truth]:
        for i in ["1", "2"]:
            d["t%s_gen_e"%i] = np.sqrt(tau_mass**2 + d["t%s_gen_px"%i]**2 + d["t%s_gen_py"%i]**2 + d["t%s_gen_pz"%i]**2)
        for k in ["px", "py", "pz", "e"]:
            d["h_gen_%s"%k] = d["t1_gen_%s"%k] + d["t2_gen_%s"%k]
        d["h_gen_pt"] = np.sqrt(d["h_gen_px"]**2 + d["h_gen_py"]**2)
        mag = np.sqrt(d["h_gen_px"]**2 + d["h_gen_py"]**2 + d["h_gen_pz"]**2)
        costheta = d["h_gen_pz"] / mag
        d["h_gen_eta"] = -0.5 * np.log((1.0 - costheta) / (1.0 + costheta))
        d["h_gen_phi"] = np.arctan2(d["h_gen_py"], d["h_gen_px"])
        d["h_gen_mass"] = np.sqrt(np.abs(d["h_gen_e"]**2 - d["h_gen_px"]**2\
                                       - d["h_gen_py"]**2 - d["h_gen_pz"]**2))

    # Plot prediction
    additionals = ["t1_gen_e", "t2_gen_e", "h_gen_px", "h_gen_py", "h_gen_pz", "h_gen_e", "h_gen_pt", "h_gen_eta", "h_gen_phi", "h_gen_mass"]
    for key in outputs_t + additionals:
        plt.figure(figsize=(6, 6))
        p = np.percentile(pred[key], [0.1, 99.9])
        q = np.percentile(truth[key], [0.1, 99.9])
        p = (min(p[0], q[0]), max(p[1], q[1]))
        _, bins, _ = plt.hist(pred[key], histtype="step", lw=3, alpha=0.8,
                              range=p, bins=20, label="NN")
        plt.hist(truth[key], histtype="step", lw=3, alpha=0.8, bins=bins, label="Truth")
        plt.legend()
        plt.xlabel(key)
        plt.xlim((bins[0], bins[-1]))
        plt.savefig(key + "_p.png")

    # Plot difference
    for key in outputs_t + additionals:
        plt.figure(figsize=(4, 4))
        diff = truth[key] - pred[key]
        p = np.percentile(diff, [0.1, 99.9])
        _, bins, _ = plt.hist(diff,
                histtype="step", lw=3, alpha=0.8, range=p, bins=30)
        plt.xlabel(key + " (true - pred)")
        plt.xlim((bins[0], bins[-1]))
        plt.savefig(key + "_diff.png")

    # Plot relative difference
    for key in outputs_t + additionals:
        plt.figure(figsize=(4, 4))
        diff = (truth[key] - pred[key])/truth[key]
        p = np.percentile(diff, [1, 99])
        _, bins, _ = plt.hist(diff,
                histtype="step", lw=3, alpha=0.8, range=p, bins=30)
        plt.xlabel(key + " (true - pred)/true")
        plt.xlim((bins[0], bins[-1]))
        plt.savefig(key + "_reldiff.png")

    # Plot mass
    def computeMass(d):
        e_t = d["h_gen_e"]
        px_t = d["h_gen_px"]
        py_t = d["h_gen_py"]
        pz_t = d["h_gen_pz"]
        return np.sqrt(np.abs(e_t**2 - px_t**2 - py_t**2 - pz_t**2))

    mass_truth = computeMass(truth)
    print("mass (truth): {:.2f} +- {:.2f}".format(np.mean(mass_truth), np.std(mass_truth)))

    mass_t = computeMass(pred)
    print("pred: {:.2f} +- {:.2f}".format(np.mean(mass_t), np.std(mass_t)))

    plt.figure(figsize=(8, 8))
    p = (0, 350)
    _, bins, _ = plt.hist(mass_t, histtype="step", lw=3, range=p, bins=30, label="pred")
    plt.hist(mass_truth, histtype="step", lw=3, bins=bins, label="truth")
    plt.legend()
    plt.xlim(p)
    plt.xlabel("mass")
    plt.savefig("mass_p.png")

    # Mass correlation
    plt.figure(figsize=(8, 8))
    p = (0, 300)
    plt.hist2d(mass_truth, mass_t, range=(p, p), bins=(100, 100), norm=mpl.colors.LogNorm())
    plt.plot(p, p, "-", color="r")
    plt.xlim(p)
    plt.ylim(p)
    plt.xlabel("truth")
    plt.ylabel("pred")
    plt.savefig("mass_t_corr_p.png")

    # Mass difference
    plt.figure(figsize=(8, 8))
    p2 = (-150, 150)
    plt.hist2d(mass_truth, mass_truth - mass_t,
            range=(p, p2), bins=(100, 100), norm=mpl.colors.LogNorm())
    plt.plot(p, [0] * len(p), "-", color="r")
    plt.xlim(p)
    plt.ylim(p2)
    plt.xlabel("truth")
    plt.ylabel("truth - pred")
    plt.savefig("mass_t_diff.png")

    # Mass rel difference
    plt.figure(figsize=(8, 8))
    p2 = (-2, 2)
    plt.hist2d(mass_truth, (mass_truth - mass_t)/mass_truth,
            bins=(100, 100), norm=mpl.colors.LogNorm())
    plt.plot(p, [0] * len(p), "-", color="r")
    plt.xlim(p)
    plt.ylim(p2)
    plt.xlabel("truth")
    plt.ylabel("(truth - pred)/truth")
    plt.savefig("mass_t_reldiff.png")

    # Plot loss and metrics
    if os.path.exists("history.pickle"):
        history = pickle.load(open(save_path+"history.pickle", "rb"))

        epochs = range(1, len(history["loss"])+1)
        for i, key in enumerate(history.keys()):
            if key.startswith("val_"):
                continue
            plt.figure(figsize=(6, 6))
            plt.plot(epochs, history[key], lw=3, label="Training loss")
            plt.plot(epochs, history["val_" + key], lw=3, label="Validation loss")
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel(key)
            plt.savefig("training_" + key + ".png")


    # Plot sample weights
    if os.path.exists("resample.pickle"):
        counts, bins, counts_after = pickle.load(open(save_path+"resample.pickle", "rb"))
        center = bins[:-1] + 0.5 * (bins[1] - bins[0])
        plt.figure(figsize=(6,6))
        plt.plot(center, counts, "o-", lw=3, label="Before")
        plt.plot(center, counts_after, "o-", lw=3, label="After")
        plt.xlabel("Count")
        plt.xlabel("Mass")
        plt.legend()
        plt.savefig("resample_mass.png")


if __name__ == "__main__":
    main()
